use rustc_abi::{Align, FIRST_VARIANT, FieldIdx, Size, VariantIdx};
use rustc_ast::Mutability;
use rustc_attr_ir::HasAttrs;
use rustc_middle::throw_machine_stop;
use rustc_middle::ty::layout::{HasTyCtxt, HasTypingEnv, ValidityRequirement};
use rustc_middle::ty::{self, FieldInfo, ScalarInt, Ty};
use rustc_span::{Symbol, bug, span_bug, sym};

use super::error::*;
use crate::const_eval::{CompileTimeMachine, MemoryKind};
use crate::interpret::{
    self, AllocInit, CtfeProvenance, Immediate, InterpCx, InterpResult, OpTy, PlaceTy, Scalar,
    ensure_monomorphic_enough, err_inval, interp_ok, throw_ub, throw_ub_format,
    type_implements_dyn_trait,
};

impl<'tcx> InterpCx<'tcx, CompileTimeMachine<'tcx>> {
    pub(crate) fn call_ctfe_intrinsic(
        &mut self,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx>],
        dest: &PlaceTy<'tcx, CtfeProvenance>,
        intrinsic_name: Symbol,
        do_fallback: impl Fn(&mut Self) -> InterpResult<'tcx, Option<ty::Instance<'tcx>>>,
    ) -> InterpResult<'tcx, Option<ty::Instance<'tcx>>> {
        match intrinsic_name {
            sym::abort_immediate => {
                // Note that `abort_immediate` is also hooked separately in Miri.
                throw_machine_stop!(ConstEvalErrKind::Abort);
            }
            sym::ptr_guaranteed_cmp => {
                let a = self.read_scalar(&args[0])?;
                let b = self.read_scalar(&args[1])?;
                let cmp = self.guaranteed_cmp(a, b)?;
                self.write_scalar(Scalar::from_u8(cmp), dest)?;
            }
            sym::const_allocate => {
                let size = self.read_scalar(&args[0])?.to_target_usize(self)?;
                let align = self.read_scalar(&args[1])?.to_target_usize(self)?;

                let align = match Align::from_bytes(align) {
                    Ok(a) => a,
                    Err(err) => {
                        throw_ub_format!("invalid align passed to `const_allocate`: {err}")
                    }
                };

                let ptr = self.allocate_ptr(
                    Size::from_bytes(size),
                    align,
                    interpret::MemoryKind::Machine(MemoryKind::Heap { was_made_global: false }),
                    AllocInit::Uninit,
                )?;
                self.write_pointer(ptr, dest)?;
            }
            sym::const_deallocate => {
                let ptr = self.read_pointer(&args[0])?;
                let size = self.read_scalar(&args[1])?.to_target_usize(self)?;
                let align = self.read_scalar(&args[2])?.to_target_usize(self)?;

                let size = Size::from_bytes(size);
                let align = match Align::from_bytes(align) {
                    Ok(a) => a,
                    Err(err) => {
                        throw_ub_format!("invalid align passed to `const_deallocate`: {err}")
                    }
                };

                // If an allocation is created in an another const,
                // we don't deallocate it.
                let (alloc_id, _, _) = self.ptr_get_alloc_id(ptr, 0)?;
                let is_allocated_in_another_const = matches!(
                    self.tcx.try_get_global_alloc(alloc_id),
                    Some(interpret::GlobalAlloc::Memory(_))
                );

                if !is_allocated_in_another_const {
                    self.deallocate_ptr(
                        ptr,
                        Some((size, align)),
                        interpret::MemoryKind::Machine(MemoryKind::Heap { was_made_global: false }),
                    )?;
                }
            }

            sym::const_make_global => {
                let ptr = self.read_pointer(&args[0])?;
                self.make_const_heap_ptr_global(ptr)?;
                self.write_pointer(ptr, dest)?;
            }

            // The intrinsic represents whether the value is known to the optimizer (LLVM).
            // We're not doing any optimizations here, so there is no optimizer that could know the value.
            // (We know the value here in the machine of course, but this is the runtime of that code,
            // not the optimization stage.)
            sym::is_val_statically_known => self.write_scalar(Scalar::from_bool(false), dest)?,

            // We handle these here since Miri does not want to have them.
            sym::assert_inhabited
            | sym::assert_zero_valid
            | sym::assert_mem_uninitialized_valid => {
                let ty = instance.args.type_at(0);
                let requirement =
                    ValidityRequirement::from_rustc_intrinsic(intrinsic_name).unwrap();

                let should_panic = !self
                    .tcx
                    .check_validity_requirement((requirement, self.typing_env().as_query_input(ty)))
                    .map_err(|_| err_inval!(TooGeneric))?;

                if should_panic {
                    let layout = self.layout_of(ty)?;

                    let msg = Symbol::intern(&match requirement {
                        // For *all* intrinsics we first check `is_uninhabited` to give a more specific
                        // error message.
                        _ if layout.is_uninhabited() => format!(
                            "aborted execution: attempted to instantiate uninhabited type `{ty}`"
                        ),
                        ValidityRequirement::Inhabited => bug!("handled earlier"),
                        ValidityRequirement::Zero => format!(
                            "aborted execution: attempted to zero-initialize type `{ty}`, which is invalid"
                        ),
                        ValidityRequirement::UninitMitigated0x01Fill => format!(
                            "aborted execution: attempted to leave type `{ty}` uninitialized, which is invalid"
                        ),
                        ValidityRequirement::Uninit => bug!("assert_uninit_valid doesn't exist"),
                    });

                    let span = self.caller_location();
                    let (file, line, col) = self.location_triple_for_span(span);

                    InterpResult::from(Err::<(), _>(ConstEvalErrKind::Panic {
                        msg,
                        file,
                        line,
                        col,
                    }))?;

                    // Skip the `return_to_block` at the end (we panicked, we do not return).
                    return interp_ok(None);
                }
            }

            sym::type_id_vtable => {
                let tp_ty = self.read_type_id(&args[0])?;
                let result_ty = self.read_type_id(&args[1])?;

                let (implements_trait, preds) = type_implements_dyn_trait(self, tp_ty, result_ty)?;

                if implements_trait {
                    let vtable_ptr = self.get_vtable_ptr(tp_ty, preds)?;
                    // Writing a non-null pointer into an `Option<NonNull>` will automatically make it `Some`.
                    self.write_pointer(vtable_ptr, dest)?;
                } else {
                    // Write `None`
                    self.write_discriminant(FIRST_VARIANT, dest)?;
                }
            }

            sym::type_id_type_of => {
                let ty = self.read_type_id(&args[0])?;
                self.write_type_info(ty, dest)?;
            }

            sym::type_id_is_signed => {
                let ty = self.read_type_id(&args[0])?;
                self.write_scalar(Scalar::from_bool(ty.is_signed()), dest)?;
            }

            sym::type_id_points_mutably => {
                let ty = self.read_type_id(&args[0])?;
                let is_mutable = matches!(
                    ty.kind(),
                    ty::RawPtr(_, Mutability::Mut) | &ty::Ref(_, _, Mutability::Mut)
                );
                self.write_scalar(Scalar::from_bool(is_mutable), dest)?;
            }

            sym::type_id_size_of => {
                let ty = self.read_type_id(&args[0])?;
                let layout = self.layout_of(ty)?;
                let variant_index = if layout.is_sized() {
                    let (variant, variant_place) = self.project_downcast_named(dest, sym::Some)?;
                    let size_field_place = self.project_field(&variant_place, FieldIdx::ZERO)?;
                    self.write_scalar(
                        ScalarInt::try_from_target_usize(layout.size.bytes(), self.tcx.tcx)
                            .unwrap(),
                        &size_field_place,
                    )?;
                    variant
                } else {
                    self.project_downcast_named(dest, sym::None)?.0
                };
                self.write_discriminant(variant_index, dest)?;
            }

            sym::type_id_element_ty => {
                let ty = self.read_type_id(&args[0])?;
                let variant_index = if let ty::Array(ty, _) | ty::Slice(ty) = ty.kind() {
                    let (variant_idx, variant_place) =
                        self.project_downcast_named(dest, sym::Some)?;
                    let type_id_field_place = self.project_field(&variant_place, FieldIdx::ZERO)?;
                    self.write_type_id(*ty, &type_id_field_place)?;
                    variant_idx
                } else {
                    self.project_downcast_named(dest, sym::None)?.0
                };
                self.write_discriminant(variant_index, dest)?;
            }

            sym::type_id_array_len => {
                let ty = self.read_type_id(&args[0])?;
                let len = if let ty::Array(_, len) = ty.kind() {
                    len.to_leaf().to_target_usize(self.tcx.tcx())
                } else {
                    0
                };
                self.write_scalar(Scalar::from_target_usize(len, self), dest)?;
            }

            sym::type_id_fields => {
                let ty = self.read_type_id(&args[0])?;
                let variant_idx = self.read_target_usize(&args[1])? as usize;

                let variants_num =
                    ty.ty_adt_def().map(|adt_def| adt_def.variants().len()).unwrap_or(1);
                if variant_idx >= variants_num {
                    throw_ub!(BoundsCheckFailed {
                        len: variants_num as u64,
                        index: variant_idx as u64
                    });
                }

                let fields_num = match ty.kind() {
                    ty::Adt(adt_def, _) => {
                        let variant_def = &adt_def.variants()[VariantIdx::from_usize(variant_idx)];
                        variant_def.fields.len()
                    }
                    ty::Tuple(fields) => fields.len(),
                    _ => 0, // Other types have no fields
                };

                self.write_scalar(Scalar::from_target_usize(fields_num as u64, self), dest)?;
            }

            sym::type_id_field_representing_type => {
                let ty = self.read_type_id(&args[0])?;
                let variant_idx = self.read_target_usize(&args[1])? as usize;
                let field_idx = self.read_target_usize(&args[2])? as usize;

                let variants_num =
                    ty.ty_adt_def().map(|adt_def| adt_def.variants().len()).unwrap_or(1);
                if variant_idx >= variants_num {
                    throw_ub!(BoundsCheckFailed {
                        len: variants_num as u64,
                        index: variant_idx as u64
                    });
                }

                let fields_num = match ty.kind() {
                    ty::Adt(adt_def, _) => {
                        let variant_def = &adt_def.variants()[VariantIdx::from_usize(variant_idx)];
                        variant_def.fields.len()
                    }
                    ty::Tuple(fields) => fields.len(),
                    _ => 0, // Other types have no fields
                };
                if field_idx >= fields_num {
                    throw_ub!(BoundsCheckFailed {
                        len: fields_num as u64,
                        index: field_idx as u64
                    });
                }

                let frt = Ty::new_field_representing_type(
                    *self.tcx,
                    ty,
                    VariantIdx::from_usize(variant_idx),
                    FieldIdx::from_usize(field_idx),
                );
                self.write_type_id(frt, dest)?;
            }
            sym::type_id_function_ptr => {
                let ty = self.read_type_id(&args[0])?;
                let variant_index = if let ty::FnPtr(sig, fn_header) = ty.kind() {
                    let (variant, variant_place) = self.project_downcast_named(dest, sym::Some)?;
                    let field_place = self.project_field(&variant_place, FieldIdx::ZERO)?;
                    let sig = sig.skip_binder(); // FIXME: handle lifetime bounds
                    self.write_fn_ptr_type_info(field_place, &sig, fn_header)?;
                    variant
                } else {
                    self.project_downcast_named(dest, sym::None)?.0
                };
                self.write_discriminant(variant_index, dest)?;
            }
            sym::type_id_points_to => {
                let ty = self.read_type_id(&args[0])?;
                let variant_index = if let ty::RawPtr(pointee_ty, _) | ty::Ref(_, pointee_ty, _) =
                    ty.kind()
                {
                    let (variant, variant_place) = self.project_downcast_named(dest, sym::Some)?;
                    let field_place = self.project_field(&variant_place, FieldIdx::ZERO)?;
                    self.write_type_id(*pointee_ty, &field_place)?;
                    variant
                } else {
                    self.project_downcast_named(dest, sym::None)?.0
                };
                self.write_discriminant(variant_index, dest)?;
            }

            sym::type_id_variants => {
                let ty = self.read_type_id(&args[0])?;
                let variants_num = ty.ty_adt_def().map(|def| def.variants().len()).unwrap_or(1);
                self.write_scalar(Scalar::from_target_usize(variants_num as u64, self), dest)?;
            }

            sym::type_id_variant_name => {
                let base = self.read_type_id(&args[0])?;

                let field_name = if let ty::Adt(def, _) = base.kind() {
                    let variant_idx = self.read_target_usize(&args[1])? as usize;
                    if variant_idx >= def.variants().len() {
                        throw_ub!(BoundsCheckFailed {
                            len: def.variants().len() as u64,
                            index: variant_idx as u64
                        });
                    }
                    let variant_idx = VariantIdx::from_usize(variant_idx);
                    def.variant(variant_idx).name
                } else {
                    span_bug!(self.cur_span(), "expected enum type, got {base}")
                };
                let ptr = self.allocate_bytes_dedup(field_name.as_str().as_bytes())?;
                self.write_immediate(
                    Immediate::ScalarPair(
                        Scalar::from_pointer(ptr, self),
                        Scalar::from_target_usize(field_name.as_str().len() as u64, self),
                    ),
                    dest,
                )?;
            }

            sym::type_id_variant_reflected_attrs => {
                let base = self.read_type_id(&args[0])?;

                if let ty::Adt(def, _) = base.kind() {
                    let variant_idx = self.read_target_usize(&args[1])? as usize;
                    if variant_idx >= def.variants().len() {
                        throw_ub!(BoundsCheckFailed {
                            len: def.variants().len() as u64,
                            index: variant_idx as u64
                        });
                    }
                    let variant_idx = VariantIdx::from_usize(variant_idx);

                    self.write_reflect_attributes(
                        dest,
                        def.variant(variant_idx).def_id.get_attrs(&self.tcx),
                    )?;
                } else {
                    span_bug!(self.cur_span(), "expected enum type, got {base}")
                };
            }

            sym::type_id_variant_non_exhaustive => {
                let base = self.read_type_id(&args[0])?;

                let non_exhaustive = if let ty::Adt(def, _) = base.kind() {
                    let variant_idx = self.read_target_usize(&args[1])? as usize;
                    if variant_idx >= def.variants().len() {
                        throw_ub!(BoundsCheckFailed {
                            len: def.variants().len() as u64,
                            index: variant_idx as u64
                        });
                    }
                    let variant_idx = VariantIdx::from_usize(variant_idx);
                    def.variant(variant_idx).is_field_list_non_exhaustive()
                } else {
                    span_bug!(self.cur_span(), "expected enum type, got {base}")
                };
                self.write_scalar(Scalar::from_bool(non_exhaustive), dest)?;
            }

            sym::field_offset => {
                let frt_ty = instance.args.type_at(0);
                ensure_monomorphic_enough(frt_ty)?;

                let (ty, variant, field) = if let ty::Adt(def, args) = frt_ty.kind()
                    && let Some(FieldInfo { base, variant_idx, field_idx, .. }) =
                        def.field_representing_type_info(self.tcx.tcx, *args)
                {
                    (base, variant_idx, field_idx)
                } else {
                    span_bug!(self.cur_span(), "expected field representing type, got {frt_ty}")
                };
                let layout = self.layout_of(ty)?;
                let cx = ty::layout::LayoutCx::new(self.tcx.tcx, self.typing_env());

                let layout = layout.for_variant(&cx, variant);
                let offset = layout.fields.offset(field.index()).bytes();

                self.write_scalar(Scalar::from_target_usize(offset, self), dest)?;
            }

            sym::field_representing_type_name => {
                let frt_ty = self.read_type_id(&args[0])?;

                let field_name = if let ty::Adt(def, args) = frt_ty.kind()
                    && let Some(FieldInfo { name, .. }) =
                        def.field_representing_type_info(self.tcx.tcx, *args)
                {
                    name
                } else {
                    span_bug!(self.cur_span(), "expected field representing type, got {frt_ty}")
                };
                let ptr = self.allocate_bytes_dedup(field_name.as_str().as_bytes())?;
                self.write_immediate(
                    Immediate::ScalarPair(
                        Scalar::from_pointer(ptr, self),
                        Scalar::from_target_usize(field_name.as_str().len() as u64, self),
                    ),
                    dest,
                )?;
            }

            sym::field_representing_type_offset => {
                let frt_ty = self.read_type_id(&args[0])?;

                let (ty, variant, field) = if let ty::Adt(def, args) = frt_ty.kind()
                    && let Some(FieldInfo { base, variant_idx, field_idx, .. }) =
                        def.field_representing_type_info(self.tcx.tcx, *args)
                {
                    (base, variant_idx, field_idx)
                } else {
                    span_bug!(self.cur_span(), "expected field representing type, got {frt_ty}")
                };
                let layout = self.layout_of(ty)?;
                let cx = ty::layout::LayoutCx::new(self.tcx.tcx, self.typing_env());

                let layout = layout.for_variant(&cx, variant);
                let offset = layout.fields.offset(field.index()).bytes();

                self.write_scalar(Scalar::from_target_usize(offset, self), dest)?;
            }

            sym::field_representing_type_actual_type_id => {
                let frt_ty = self.read_type_id(&args[0])?;

                let field_ty = if let ty::Adt(def, args) = frt_ty.kind()
                    && let Some(FieldInfo { ty, .. }) =
                        def.field_representing_type_info(self.tcx.tcx, *args)
                {
                    self.tcx.erase_and_anonymize_regions(ty)
                } else {
                    span_bug!(self.cur_span(), "expected field representing type, got {frt_ty}")
                };
                self.write_type_id(field_ty, dest)?;
            }

            sym::type_id_reflected_attrs => {
                let ty = self.read_type_id(&args[0])?;
                let attrs = match ty.kind() {
                    ty::Adt(def, args)
                        if let Some(FieldInfo { base, variant_idx, field_idx, .. }) =
                            def.field_representing_type_info(self.tcx(), *args) =>
                    {
                        match base.kind() {
                            ty::Adt(base_def, _) => {
                                let field = &base_def.variants()[variant_idx].fields[field_idx];
                                field.did.get_attrs(&self.tcx)
                            }
                            ty::Tuple(_) => &[],
                            _ => {
                                // only adts and tuples have fields
                                span_bug!(self.cur_span(), "expected adt or tuple type, got {ty}");
                            }
                        }
                    }
                    ty::Adt(def, _) => def.did().get_attrs(&self.tcx),
                    _ => &[],
                };

                self.write_reflect_attributes(dest, attrs)?;
            }

            sym::type_id_generics => {
                let ty = self.read_type_id(&args[0])?;
                self.write_type_id_generics(dest, ty)?;
            }

            sym::type_id_non_exhaustive => {
                let ty = self.read_type_id(&args[0])?;

                // FIXME(reflection): need a way to obtain non-exhaustiveness of a variant's fields.
                let non_exhaustive = if let ty::Adt(def, _) = ty.kind() {
                    if def.is_enum() {
                        def.is_variant_list_non_exhaustive()
                    } else {
                        def.non_enum_variant().is_field_list_non_exhaustive()
                    }
                } else {
                    false
                };

                self.write_scalar(Scalar::from_bool(non_exhaustive), dest)?;
            }

            _ => {
                return do_fallback(self);
            }
        }

        interp_ok(None)
    }
}
