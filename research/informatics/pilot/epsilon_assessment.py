"""Expose unsuccessful native 3-6-9 candidates rather than hiding Epsilon."""
def assessment(d):
 candidates=[]
 for f in d['formation_catalogue']:
  if f['shape']!='Hexagon':continue
  values=f['native_values'];missing=sorted({3,6,9}-set(values));failed=[{'alpha_index':i,'clock_position':i or d['pisano_period'],'value':v} for i,v in zip(f['alpha_indices'],values) if v not in (3,6,9)]
  candidates.append({'formation_id':f['id'],'alpha_indices':f['alpha_indices'],'clock_positions':f['clock_positions'],'native_values':values,'rotation_degrees':f['rotation_degrees'],'missing_required_values':missing,'failing_nodes':failed,'native_369_qualifies':bool(f.get('epsilon_routes'))})
 return {'rule_version':'native-epsilon-diagnostics-v1','native_369_present':bool(d['epsilon']),'candidate_count':len(candidates),'candidates':candidates,'absence_reason':None if d['epsilon'] else 'No equally spaced six-node formation exists on this period circle.' if not candidates else 'No tested six-node formation contains only 3, 6 and 9 with all three present.','reference_path_available':bool(d.get('epsilon_path'))}
