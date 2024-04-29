import numpy as np
from typing import Tuple
from pm4py.objects.log.obj import Trace

def is_compliant_fines(trace: Trace) -> Tuple[str, str]:
    compliance = "compliant"
    compliance_type = "compliant"
    if trace[-1]["concept:name"] == "Send Appeal to Prefecture":
        compliance = "incompliant"
        compliance_type = "incompliant-SA"
    elif trace[-1]["concept:name"] == "Receive Result Appeal from Prefecture":
        compliance = "incompliant"
        compliance_type = "incompliant-RR"
    # elif trace[-1]["concept:name"] == "Notify Result Appeal to Offender":
    #     compliance = "incompliant"
    #     compliance_type = "incompliant-NR"
    # elif trace[-1]["concept:name"] == "Appeal to Judge":
    #     compliance = "incompliant"
    #     compliance_type = "incompliant-AJ"
    return compliance, compliance_type

def is_compliant_incomplete_bpic2019(trace: Trace) -> Tuple[str, str]:
    compliance = "incompliant"
    completeness = "complete"
    gr_number = 0
    cgr_number = 0
    ci_number = 0
    ir_number = 0
    cir_number = 0
    net_value = np.round(trace[0]['Cumulative net worth (EUR)'], 2)
    cumulated_value_gr = 0
    cumulated_value_ir = 0
    item_category = trace.attributes['Item Category']
    for event in trace:
        if event['concept:name'] == 'Record Goods Receipt':
            gr_number += 1
            cumulated_value_gr += np.round(event['Cumulative net worth (EUR)'], 2)
        if event['concept:name'] == 'Record Invoice Receipt':
            ir_number += 1
            cumulated_value_ir += np.round(event['Cumulative net worth (EUR)'], 2)
        if event['concept:name'] == 'Clear Invoice':
            ci_number += 1
        if event['concept:name'] == 'Cancel Invoice Receipt':
            cir_number += 1
        if event['concept:name'] == 'Cancel Goods Receipt':
            cgr_number += 1
    case_properties = {
            'good_receipt': gr_number, 'clear_invoice': ci_number,
            'invoice_receipt': ir_number, 'net_value': net_value,
            'cumulated_value_good_receipt': cumulated_value_gr,
            'cumulated_value_invoice_receipt': cumulated_value_ir,
            'item_category': item_category, 'end_activity': trace[-1]['concept:name'],
            'cancel_good_receipt': cgr_number, 'cancel_invoice_receipt': cir_number,
            }
    if item_category == '3-way match, invoice after GR':
        compliance, completeness = three_way_invoice_after_gr(case_properties)
    elif item_category == '3-way match, invoice before GR':
        compliance, completeness = three_way_invoice_before_gr(case_properties)
        #if compliance == 'incompliant' and completeness == 'complete':
        #    breakpoint()
    elif item_category == '2-way match':
        compliance, completeness = two_way(case_properties)
    elif item_category == 'Consignment':
        #breakpoint()
        compliance, completeness = consigment(case_properties)
    else:
        raise ValueError("item category not recognised")
    return compliance, completeness

def three_way_invoice_after_gr(case_properties: dict) -> Tuple[str, str]:
    compliance = "incompliant"
    completeness = "complete"
    first_rule = (case_properties['clear_invoice'] > 0) and (case_properties['invoice_receipt'] > 0) and (case_properties['good_receipt'] > 0) 
    cgr_num = case_properties['cancel_good_receipt']
    cir_num = case_properties['cancel_invoice_receipt']
    tot_gr = case_properties['good_receipt'] - cgr_num
    tot_ir = case_properties['invoice_receipt'] - cir_num
    if not first_rule:
        if not case_properties['end_activity'] == 'Clear Invoice' or tot_ir < tot_gr:
            completeness = "incomplete"
        return compliance, completeness
    second_rule = np.round(case_properties['net_value'], 2) == np.round((case_properties['cumulated_value_invoice_receipt'] / case_properties['invoice_receipt']), 2)
    third_rule = np.round(case_properties['cumulated_value_invoice_receipt'], 2) == np.round(case_properties['cumulated_value_good_receipt'], 2)
    fourth_rule = case_properties['good_receipt'] == case_properties['invoice_receipt']
    fifth_rule = case_properties['end_activity'] == 'Clear Invoice'
    if tot_ir < tot_gr:
        completeness = "incomplete"
        return compliance, completeness
    if second_rule and third_rule and fourth_rule:
        compliance = "compliant"
    if not fifth_rule:
        compliance = "incompliant"
        completeness = "incomplete"
    return compliance, completeness

def three_way_invoice_before_gr(case_properties: dict) -> Tuple[str, str]:
    compliance = "incompliant"
    completeness = "complete"
    first_rule = (case_properties['clear_invoice'] > 0) and (case_properties['invoice_receipt'] > 0) and (case_properties['good_receipt'] > 0) 
    cgr_num = case_properties['cancel_good_receipt']
    cir_num = case_properties['cancel_invoice_receipt']
    tot_gr = case_properties['good_receipt'] - cgr_num
    tot_ir = case_properties['invoice_receipt'] - cir_num
    if not first_rule:
        if tot_ir != tot_gr or not case_properties['end_activity'] == 'Clear Invoice':
            completeness = "incomplete"
        return compliance, completeness
    if tot_ir != tot_gr or not case_properties['end_activity'] == 'Clear Invoice':
        completeness = "incomplete"
        return compliance, completeness 
    second_rule = np.round(case_properties['net_value'], 2) == np.round((case_properties['cumulated_value_invoice_receipt'] / case_properties['invoice_receipt']), 2)
    third_rule = np.round(case_properties['cumulated_value_invoice_receipt'], 2) == np.round(case_properties['cumulated_value_good_receipt'], 2)
    fourth_rule = case_properties['good_receipt'] == case_properties['invoice_receipt']
    if second_rule and third_rule and fourth_rule:
        compliance = "compliant"
    return compliance, completeness

def two_way(case_properties: dict) -> Tuple[str, str]:
    compliance = "incompliant"
    completeness = "complete"
    if not case_properties['end_activity'] == 'Clear Invoice':
        completeness = "incomplete"
        return compliance, completeness
    if (case_properties['clear_invoice'] > 0) and (case_properties['invoice_receipt']) > 0:
        compliance = "compliant"
    return compliance, completeness

def consigment(case_properties: dict) -> Tuple[str, str]:
    compliance = "incompliant"
    completeness = "complete"
    if not case_properties['end_activity'] == 'Record Goods Receipt':
        completeness = "incomplete"
        return compliance, completeness
    if not case_properties['good_receipt'] > 0:
        if not case_properties['end_activity'] == 'Record Goods Receipt':
            completeness = "incomplete"
        return compliance, completeness
    second_rule = np.round(case_properties['net_value'], 2) == np.round((case_properties['cumulated_value_good_receipt'] / case_properties['good_receipt']), 2)
    if second_rule:
        compliance = "compliant"
    return compliance, completeness
