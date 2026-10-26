// Report identity and execution receipts are checked before anything is labelled a result.
export function verifiedReport(report,study,kind){
 if(!report||!study||report.state!=='complete'||report.kind!==kind||report.study_id!==study.id||report.study_version!==study.version||typeof report.finding!=='string'||!report.finding.trim())return null;
 const r=report.receipt;if(!r||typeof r.run_id!=='string'||!r.run_id||typeof r.model_version!=='string'||!r.model_version||!Number.isFinite(Date.parse(r.completed_at))||!/^[a-f0-9]{64}$/i.test(r.input_sha256||''))return null;
 const allowed={classical:['classical-cpu','classical-gpu'],simulation:['cudaq-cpu','cudaq-gpu'],quantum:['aquila-qpu','dwave-qpu']};
 if(!allowed[kind]?.includes(r.execution))return null;
 if((kind==='quantum'||r.execution==='cudaq-gpu')&&(!r.task_id||typeof r.task_id!=='string'))return null;
 return report;
}
export function reportPair(classical,quantum){return !!(classical&&quantum&&classical.study_id===quantum.study_id&&classical.study_version===quantum.study_version&&classical.receipt.input_sha256===quantum.receipt.input_sha256);}
export const timing={classical:'Approximately 5–10 minutes',quantum:'Timing varies with the quantum queue'};
export const sharedStudy={id:'oxtr-qrc-01',version:'1.0.0'};
export function reportContext(active,state,pending=false){
 if(pending)return {study:null,reports:{},matched:false,shared:false};
 const study=active?active.study:sharedStudy;
 const matched=!!(study&&state?.study_id===study.id&&state.study_version===study.version);
 return {study,reports:matched?state.reports||{}:active?.reports||{},matched,shared:!active&&matched};
}
