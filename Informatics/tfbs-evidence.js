// Apply a source-linked interpretation without inferring outcomes from a test label.
export function bindingEvidence(catalogue, source, record) {
  const guide = catalogue.binding_interpretation;
  if (!guide) throw Error('Binding-outcome definitions unavailable');
  const key = record.study_id === 'braintf' ? 'braintf' : source === 'published' ? 'studies' : source;
  const evidence = guide.records[record.id] || guide.defaults[key] || guide.defaults.studies;
  return {...evidence, interpretation_version:guide.version};
}

export function withBindingEvidence(catalogue, source, record) {
  return {...record, binding_evidence:bindingEvidence(catalogue, source, record)};
}
