from collections import OrderedDict

from preprocess.parse_csv import EHRParser


def encode_concept(patient_admission, admission_diagnoses, admission_procedures=None):
    """
    Dual-stream version: Encode diagnoses and procedures separately
    If admission_procedures is None, works as original single-stream
    """
    
    # Xử lý dual-stream: có procedures
    if admission_procedures is not None:
        # Tạo map RIÊNG cho diagnoses và procedures
        diagnosis_map = OrderedDict()
        procedure_map = OrderedDict()
        
        print("Encoding diagnoses and procedures separately...")
        
        # Encode diagnoses
        for pid, admissions in patient_admission.items():
            for admission in admissions:
                adm_id = admission[EHRParser.adm_id_col]
                if adm_id in admission_diagnoses:
                    diagnoses = admission_diagnoses[adm_id]
                    for diagnosis in diagnoses:
                        if diagnosis not in diagnosis_map:
                            diagnosis_map[diagnosis] = len(diagnosis_map)
        
        # Encode procedures  
        for pid, admissions in patient_admission.items():
            for admission in admissions:
                adm_id = admission[EHRParser.adm_id_col]
                if adm_id in admission_procedures:
                    procedures = admission_procedures[adm_id]
                    for procedure in procedures:
                        if procedure not in procedure_map:
                            procedure_map[procedure] = len(procedure_map)

        print(f"Encoded {len(diagnosis_map)} unique diagnosis codes")
        print(f"Encoded {len(procedure_map)} unique procedure codes")
        
        # Encode diagnoses data
        admission_diagnoses_encoded = {
            admission_id: list(OrderedDict.fromkeys(diagnosis_map[diagnosis] for diagnosis in diagnoses))
            for admission_id, diagnoses in admission_diagnoses.items()
            if admission_id in admission_diagnoses
        }
        
        # Encode procedures data
        admission_procedures_encoded = {
            admission_id: list(OrderedDict.fromkeys(procedure_map[procedure] for procedure in procedures))
            for admission_id, procedures in admission_procedures.items()
            if admission_id in admission_procedures
        }

        # Thống kê
        total_encoded_diag = sum(len(diags) for diags in admission_diagnoses_encoded.values())
        total_encoded_proc = sum(len(procs) for procs in admission_procedures_encoded.values())
        print(f"Total encoded diagnosis instances: {total_encoded_diag}")
        print(f"Total encoded procedure instances: {total_encoded_proc}")
        
        return admission_diagnoses_encoded, diagnosis_map, admission_procedures_encoded, procedure_map
    
    # Xử lý single-stream (backward compatibility)
    else:
        concept_map = OrderedDict()
        print("Encoding concepts (single-stream mode)...")
        
        for pid, admissions in patient_admission.items():
            for admission in admissions:
                adm_id = admission[EHRParser.adm_id_col]
                if adm_id in admission_diagnoses:  # admission_diagnoses thực chất là admission_concepts
                    concepts = admission_diagnoses[adm_id]
                    for concept in concepts:
                        if concept not in concept_map:
                            concept_map[concept] = len(concept_map)

        admission_concept_encoded = {
            admission_id: list(OrderedDict.fromkeys(concept_map[concept] for concept in concepts))
            for admission_id, concepts in admission_diagnoses.items()
        }
        
        print(f"Encoded {len(concept_map)} unique codes")
        print(f"Total encoded instances: {sum(len(concepts) for concepts in admission_concept_encoded.values())}")
        
        return admission_concept_encoded, concept_map