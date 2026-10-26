"""Publish an explicit qualification record from receipts, never a predictive claim."""
from pathlib import Path
import json, hashlib
import math
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results'
SITE = Path('/private/tmp/um-quantum-lab-20261025/Informatics/quantum/data')
def read(name): return json.loads((OUT/name).read_text())
def save(path,data): path.write_text(json.dumps(data,indent=2)+'\n')
local, gpu, hw = read('qualification.json'), read('gpu-qualification.json'), read('hardware-qualification.json')
assert local['complete'] and gpu['complete'] and len(hw['records'])==16
inputs=read('inputs.json'); physics=read('physical-model.json'); wolfram=read('wolfram-comparison.json')
ih=hashlib.sha256((OUT/'inputs.json').read_bytes()).hexdigest()
records=[]
for r in local['records']:
 g=next(x for x in gpu['cases'] if x['id']==r['id'])
 h=next(x for x in hw['records'] if x['id']==r['id'])
 records.append({'id':r['id'],'window':r['window'],'variant':r['variant'],'ideal_observables':r['ideal_observables'],
                 'gpu_observables':g['measured_observables'],'gpu_counts':g['counts'],'shots':g['shots'],
                 'gpu_job_id':g['job_id'],'hardware_task_id':h['task_arn'].rsplit('/',1)[1],
                 'hardware_state':h['status'],'qasm_sha256':r['qasm_sha256'],
                 **({'hardware_observables':h['measured_observables'],'hardware_counts':h['counts'],'hardware_valid_shots':h['fully_occupied_shots']} if h.get('measured_observables') else {})})
artifact={'schema':'universal-informatics/qualification-results/v1','study_id':'oxtr-qrc-01','study_version':'1.0.0','model_version':'1.1.0-qualification',
          'updated_at':hw['at'],'purpose':'Two-window measurement qualification. Occupancy prediction is not yet evaluated.',
          'input_sha256':ih,'positions_m':physics['positions_m'],'windows':[{k:r[k] for k in ['id','start','end']} for r in inputs['rows']],
          'records':records,'gpu_cost_credits':gpu['cost_credits'],'hardware_fee_usd':20.80,
          'assay_candidate':'ENCODE ENCSR932ZMX · CTCF · GM23338','local_convergence_max_error':max(r['convergence'][-1]['max_observable_error'] for r in local['records']),
          'wolfram_reference_max_error':wolfram['maximum_probability_error']}
save(SITE/'oxtr-qrc-qualification.json',artifact)
def receipt(run,execution,backend,at,task=None):
 r={'run_id':run,'model_version':'1.1.0-qualification','execution':execution,'backend':backend,'completed_at':at,'input_sha256':ih}
 if task:r['task_id']=task
 return r
def report(kind,title,finding,receipt,metrics,details):
 return {'state':'complete','kind':kind,'study_id':'oxtr-qrc-01','study_version':'1.0.0','title':title,'finding':finding,'receipt':receipt,'metrics':metrics,'details':details,'phase':'qualification'}
classical=report('classical','The mathematical model passed its checks.',
 'All 16 programs passed numerical comparison. Wolfram Quantum Framework independently matched the reference case. This establishes a consistent calculation for the first measurement phase; it does not yet establish improved prediction of OXTR occupancy.',
 receipt('oxtr-qrc-01-local-qualification','classical-cpu','SciPy reference + CUDA-Q CPU + Wolfram Quantum Framework',local['at']),
 [{'label':'Model cases checked','value':16},{'label':'Maximum final numerical difference','value':'0.0000583 in probability'},{'label':'Wolfram reference difference','value':'0.00000000726 in probability'}],
 ['Six atoms, full 64-state evolution, all 15 pair interactions. Same coordinates, pulses and interactions in the GPU and Aquila programs.',
  'Two motif-rich 256-base OXTR windows were selected without occupancy labels. They qualify execution and are not an independent prediction test.',
  'Native Fibonacci, Phi Ratio, Alpha, Beta, Gamma and Epsilon are compared with ordinary and shuffled inputs.',
  'Gamma and Epsilon produce distinct simulated responses. At 100 shots, measurement uncertainty can be larger than that difference.'])
simulation=report('simulation','All 16 GPU simulations are complete.',
 'The CUDA-Q programs ran on the NVIDIA GPU through qBraid. All sampled responses passed the prespecified numerical agreement check. Explore how each representation changes the response of the six-atom model.',
 receipt('oxtr-qrc-01-gpu-qualification','cudaq-gpu','CUDA-Q compiled programs → qBraid NVIDIA GPU',gpu['at'],gpu['cases'][1]['job_id']),
 [{'label':'Study simulations','value':'16 × 100 shots'},{'label':'Ordering check','value':'Passed · 8 shots'},{'label':'GPU charge','value':'0 credits'}],
 ['The circles show simulated excitation probabilities, not protein binding probabilities. Every point includes 100-shot sampling uncertainty.',
  'The acceptance check uses six binomial standard errors plus two counts; it is an execution check, not a predictive-performance result.',
  'No classifier was trained in this qualification batch. OXTR occupancy evaluation remains a separate phase using held-out biological evidence.'])
simulation['response_data']='oxtr-qrc-qualification.json'
def mean_response(values): return sum(values[:6])/6
window_ids=list(dict.fromkeys(r['window'] for r in records))
def delta(window,variant,field):
 arm=next(r for r in records if r['window']==window and r['variant']==variant)
 control=next(r for r in records if r['window']==window and r['variant']=='ordinary')
 return 100*(mean_response(arm[field])-mean_response(control[field]))
def changes(variant,field='ideal_observables'):
 return ' and '.join(f'{delta(w,variant,field):+.1f}' for w in window_ids)
classical['title']='The Fibonacci representations change the response to the same OXTR DNA.'
classical['finding']='Holding the DNA window and six-atom model fixed, changing its mathematical representation changes the calculated quantum response. The effects differ between the two OXTR windows: the representations do not act as one interchangeable Fibonacci pattern.'
classical['insights']=[
 {'title':'What this tells us about the occupancy question','text':'We have generated distinct candidate features from OXTR DNA. This batch did not compare them with measured transcription-factor occupancy, so improved prediction has not yet been established.'},
 {'title':'The contribution of this first experiment','text':'It identifies which representations alter the calculated response and tests whether a physical quantum processor reproduces those changes. Those are concrete properties to examine before interpreting a response as biological information.'}]
simulation['title']='Gamma and Epsilon increase the simulated response in both DNA windows.'
simulation['finding']='Among the six proposed representations, Gamma and Epsilon produce the largest increases over ordinary input in the ideal model. The GPU samples accompany that reference calculation below. A larger atom response is not a measure of better binding-site prediction.'
simulation['insights']=[
 {'title':'Native Fibonacci and Phi Ratio','text':f'Native Fibonacci changes the ideal response by {changes("fibonacci")} percentage points in windows 1 and 2. Phi Ratio changes it by {changes("phi")}. Their effect is much smaller in the second window.'},
 {'title':'Alpha and Beta','text':f'Both reduce the ideal response in both windows: Alpha {changes("alpha")} percentage points; Beta {changes("beta")}. The individual M9 P24 paths generate different responses.'},
 {'title':'Gamma and Epsilon','text':f'Gamma increases the ideal response by {changes("gamma")} percentage points; Epsilon by {changes("epsilon")}. Their different path ordering remains visible even though they use the same six period positions.'},
 {'title':'Why the shuffled control matters','text':f'The shuffled representation increases the second-window response by {delta(window_ids[1],"shuffled","ideal_observables"):.1f} percentage points, more than any proposed Fibonacci representation. Response size alone cannot identify a specific Fibonacci benefit.'}]
state=json.loads((SITE/'oxtr-qrc-state.json').read_text())
state.update(state='measurement_phase',updated_at=hw['at'],updated_on=hw['at'][:10],phase='qualification',
             status_label='First measurement phase',caption='16 GPU simulations complete. 16 Aquila tasks accepted. OXTR occupancy prediction remains to be evaluated.')
state['stages'].update({
 'model':{'state':'complete','label':'Model checks passed','message':'All 16 programs passed numerical checks. Wolfram Quantum Framework independently verified the reference Hamiltonian and evolution.'},
 'classical':{'state':'complete','label':'Numerical checks complete'},
 'simulation':{'state':'complete','label':'16 GPU runs complete','message':'All eight representations were simulated for two OXTR windows. CUDA-Q compiled programs ran through qBraid on an NVIDIA GPU at zero credit cost.'},
 'hardware':{'state':'queued','label':'16 Aquila tasks accepted','message':'The 16 matched programs have Amazon Braket task receipts. Each requests 100 measurements. Results appear after hardware processing and comparison.'},
 'evidence':{'state':'specified','label':'Prediction evaluation follows','message':'This first batch checks the measurement route. The larger benchmark compares held-out occupancy predictions against strong classical baselines.'},
 'result':{'state':'planned','label':'Simulation checks available','message':'The numerical and GPU qualification results are available under Classical results. Biological prediction has not yet been evaluated.'}})
state['jobs']=[{'provider':'qBraid','id':g['job_id'],'state':g['status']} for g in gpu['cases']]+[{'provider':'Amazon Braket / QuEra','id':h['task_arn'].rsplit('/',1)[1],'state':h['status']} for h in hw['records']]
completed=sum(h['status']=='COMPLETED' for h in hw['records'])
running=sum(h['status']=='RUNNING' for h in hw['records'])
queued=sum(h['status']=='QUEUED' for h in hw['records'])
failed=sum(h['status'] in ['FAILED','CANCELLED'] for h in hw['records'])
state['stages']['hardware'].update(state='failed' if failed else 'complete' if completed==16 else 'running' if running else 'queued',
 label=f'{completed}/16 returned · {running} running' if running else f'{completed}/16 returned · {queued} queued',
 message=f'Recorded at {hw["at"]}: {completed} completed, {running} running, {queued} queued, {failed} interrupted. All 16 tasks have saved receipts. The comparison is released after the full batch is evaluated.')
state['caption']=f'16 GPU simulations complete. Aquila: {completed} completed, {running} running, {queued} queued. Status recorded {hw["at"]}. Occupancy prediction remains to be evaluated.'
state['reports'].update(classical=classical,simulation=simulation)
if hw['complete']:
 assert completed==16 and all(h.get('fully_occupied_shots',0)>0 for h in hw['records'])
 for h in hw['records']:
  assert h['returned_shots']==100 and sum(h['counts'].values())==h['fully_occupied_shots']
  assert h['failed_shots']+h['initially_empty_shots']+h['fully_occupied_shots']==100
  assert len(h['measured_observables'])==21 and all(math.isfinite(v) and 0<=v<=1 for v in h['measured_observables'])
 errors=[abs(a-b) for h in hw['records'] for a,b in zip(h['measured_observables'],h['ideal_observables'])]
 mean_error=sum(errors)/len(errors)
 valid=sum(h['fully_occupied_shots'] for h in hw['records'])
 excluded=sum(h['initially_empty_shots'] for h in hw['records'])
 failed_shots=sum(h['failed_shots'] for h in hw['records'])
 completed_at=max(h['ended_at'] for h in hw['records'])
 comparison={'mean_absolute_observable_difference':mean_error,'maximum_absolute_observable_difference':max(errors),
             'programs':16,'observables_per_program':21,'requested_shots':1600,'returned_shots':1600,
             'fully_occupied_shots':valid,'initially_empty_shots':excluded,'failed_shots':failed_shots,
             'interpretation':'Descriptive measurement comparison against the ideal reference; no occupancy prediction or quantum advantage estimate.'}
 artifact['hardware_comparison']=comparison
 save(SITE/'oxtr-qrc-qualification.json',artifact)
 quantum=report('quantum','All 16 Aquila measurements are complete.',
  f'QuEra Aquila returned 1,600 measurements across both OXTR windows and all eight representations. {valid:,} measurements began with all six atoms present. Explore the recorded quantum responses alongside their mathematical reference.',
  receipt('oxtr-qrc-01-aquila-qualification','aquila-qpu','QuEra Aquila via Amazon Braket',completed_at,hw['records'][0]['task_arn'].rsplit('/',1)[1]),
  [{'label':'Quantum programs','value':'16 × 100 shots'},{'label':'Fully occupied measurements','value':f'{valid:,} / 1,600'},
   {'label':'Average difference from reference','value':f'{mean_error*100:.2f} percentage points'}],
  [f'The average absolute difference covers all 21 observables in each of 16 programs. It is descriptive, not an occupancy-prediction score.',
   f'{excluded} measurements began with at least one atom missing and are excluded from the conditional response. {failed_shots} measurements were reported as failed.',
   'An atom absent at readout can indicate excitation or loss. The displayed 95% intervals cover sampling uncertainty and do not include every hardware error.',
   'These completed measurements qualify the execution route. Whether the representations improve OXTR occupancy prediction remains an independent biological test.'])
 quantum['response_data']='oxtr-qrc-qualification.json'
 quantum['title']='Aquila reproduces some changes—and reveals a mismatch for Gamma and Epsilon.'
 quantum['finding']='Alpha and Beta reduce the observed response in both OXTR windows, matching the direction predicted by the model. Gamma and Epsilon increase it in the first window but decrease it in the second. The hardware therefore does not reproduce one consistent ordering of the representations across both DNA windows.'
 quantum['insights']=[
  {'title':'Native Fibonacci and Phi Ratio','text':f'Native Fibonacci changes the measured response by {changes("fibonacci","hardware_observables")} percentage points in windows 1 and 2. Phi Ratio changes it by {changes("phi","hardware_observables")}. These are observed differences from ordinary input, not changes in biological binding probability.'},
  {'title':'Alpha and Beta','text':f'Alpha changes the measured response by {changes("alpha","hardware_observables")} percentage points; Beta by {changes("beta","hardware_observables")}. Their downward direction matches the model in both windows. This small batch does not establish that the pattern repeats reliably.'},
  {'title':'Gamma and Epsilon','text':f'Gamma changes the measured response by {changes("gamma","hardware_observables")} percentage points; Epsilon by {changes("epsilon","hardware_observables")}. The second-window reversal shows that the ideal model does not fully describe these hardware measurements.'},
  {'title':'The control and the OXTR question','text':f'The shuffled control has the largest measured response in window 2 ({delta(window_ids[1],"shuffled","hardware_observables"):+.1f} percentage points). We have measured representation-dependent responses, but have not established a Fibonacci-specific predictive benefit. Occupancy labels were not tested in this batch.'}]
 quantum['details'].append('All contrasts are exploratory descriptions. There is one hardware task per representation and window, run in a fixed order. Sampling uncertainty, drift and systematic errors limit causal or repeatability claims; multiple comparisons have not been used to declare a significant winner.')
 state['reports'].update(quantum=quantum,comparison={'state':'complete','classical_run_id':classical['receipt']['run_id'],
  'quantum_run_id':quantum['receipt']['run_id'],
  'finding':'The first experiment shows that Fibonacci, Phi Ratio and the four M9 P24 paths generate different calculated responses to OXTR DNA. Aquila reproduces some directional changes but not all; the shuffled control also generates a strong response. This gives us candidate quantum features and a concrete simulation–hardware mismatch to investigate. It does not yet answer whether those features improve transcription-factor occupancy prediction: no occupancy predictor or held-out biological test was run.'})
 state.update(state='qualification_complete',status_label='First measurement phase complete',
              caption='16 GPU simulations and 16 Aquila programs complete. Explore both sets of responses. OXTR occupancy prediction remains to be evaluated.')
 state['stages']['hardware'].update(state='complete',label='16 Aquila runs complete',message=f'All 16 tasks returned 100 measurements each. {valid:,} measurements began with all six atoms present. The recorded quantum comparison is available under Quantum results.')
 state['stages']['result'].update(state='complete',label='Measurement comparison ready',message='Classical, GPU and Aquila measurement results are available. This is the first measurement phase; the biological prediction question remains open.')
save(SITE/'oxtr-qrc-state.json',state)
study=json.loads((SITE/'oxtr-qrc-study.json').read_text())
study['status']='measurement_qualification'
study['scope']='The first measurement phase has completed CPU and GPU checks and has submitted 16 Aquila programs. Held-out occupancy prediction remains to be evaluated.'
study['execution']['current_stage']='hardware'
study['execution']['job_records']=state['jobs']
study['qualification']={'version':'1.1.0-qualification','status':'hardware_submitted','report':'oxtr-qrc-qualification.json','methods':'oxtr-qrc-qualification.md',
                        'windows':2,'representations':8,'shots_per_program':100,'scope':'Execution and numerical qualification; no biological prediction result.'}
study['review']={'status':'Measurement phase activated','next':'Retrieve Aquila measurements and compare them with the matched simulations before the held-out prediction benchmark.','methods_url':'data/oxtr-qrc-protocol.md'}
study['delivery']['phase']='measurement_qualification'
if hw['complete']:
 study['status']='measurement_qualification_complete'
 study['scope']='The first measurement phase has completed CPU and GPU checks and all 16 Aquila programs. Held-out occupancy prediction remains to be evaluated.'
 study['execution']['current_stage']='qualification_complete'
 study['qualification']['status']='complete'
 study['review']={'status':'First measurement phase complete','next':'Evaluate the prespecified prediction benchmark against held-out biological observations.','methods_url':'data/oxtr-qrc-qualification.md'}
save(SITE/'oxtr-qrc-study.json',study)
print('Published data prepared from 17 GPU and 16 Aquila receipts.')
