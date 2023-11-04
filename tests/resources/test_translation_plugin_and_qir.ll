entry:
  call void @__quantum__rt__initialize(i8* null)
  call void @__quantum__qis__prepare_x__body(i1 false, %Qubit* null)
  call void @__quantum__qis__prepare_z__body(i1 true, %Qubit* inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__prepare_z__body(i1 false, %Qubit* inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__prepare_x__body(i1 true, %Qubit* inttoptr (i64 3 to %Qubit*))
  call void @__quantum__qis__mx__body(%Qubit* inttoptr (i64 3 to %Qubit*), %Result* inttoptr (i64 1 to %Result*))
  call void @__quantum__qis__mz__body(%Qubit* null, %Result* inttoptr (i64 2 to %Result*))
  %0 = call i1 @__quantum__qis__read_result__body(%Result* inttoptr (i64 2 to %Result*))
  br i1 %0, label %then, label %else

then:                                             ; preds = %entry
  call void @__quantum__qis__x__body(%Qubit* inttoptr (i64 2 to %Qubit*))
  br label %continue

else:                                             ; preds = %entry
  br label %continue

continue:                                         ; preds = %else, %then
  call void @__quantum__qis__mx__body(%Qubit* inttoptr (i64 2 to %Qubit*), %Result* null)
  call void @__quantum__qis__mz__body(%Qubit* inttoptr (i64 1 to %Qubit*), %Result* inttoptr (i64 3 to %Result*))
  call void @__quantum__rt__array_record_output(i64 4, i8* null)
  call void @__quantum__rt__result_record_output(%Result* inttoptr (i64 3 to %Result*), i8* null)
  call void @__quantum__rt__result_record_output(%Result* inttoptr (i64 2 to %Result*), i8* null)
  call void @__quantum__rt__result_record_output(%Result* inttoptr (i64 1 to %Result*), i8* null)
  call void @__quantum__rt__result_record_output(%Result* null, i8* null)
  ret void
