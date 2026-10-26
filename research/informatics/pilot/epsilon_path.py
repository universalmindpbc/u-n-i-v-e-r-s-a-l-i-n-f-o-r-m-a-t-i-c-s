"""Exact north-oriented M9 reference-path comparison, separate from native qualification."""
REFERENCE=[9,3,6,9,3,6,9]
TURNS=[0,2,5,3,1,4,0]
def path_comparison(d):
 m,p=d['modulus'],d['pisano_period']; geometry=p%6==0
 route=[p*k//6 for k in TURNS] if geometry else []
 native=[d['alpha']['display_values'][i] for i in route]
 reduced=[d['alpha']['residues'][i]%9 or 9 for i in route]
 numeric=geometry and reduced==REFERENCE
 return {'rule_version':'epsilon-m9-exact-v2','geometry_identical':geometry,'reduced_sequence_identical':numeric,'combined_eligible':geometry and numeric,'alpha_indices':route,'clock_positions':[i or p for i in route],'native_values':native,'reduced_m9_values':reduced,'reference_values':REFERENCE,'normalized_route_sixths':TURNS,'orientation':'clockwise from north; fixed M9 route and starting phase','reason':'Exact geometry and reduced traversal match.' if numeric else 'No regular six-node formation on this period circle.' if not geometry else 'Reference geometry exists, but the reduced traversal does not equal 9–3–6–9–3–6–9.','scope':'Tests the fixed north-oriented M9 path on the full native period circle; does not fold the period or rotate/reverse the values to force a match.'}
def reference_path(d):
 c=path_comparison(d)
 if not c['combined_eligible'] or d.get('epsilon'):return None
 m=d['modulus']
 return {**c,'reference':'M9 normalized circle geometry and exact reduced traversal','geometry_present':True,'native_369_qualifies':set(c['native_values'])=={3,6,9},'reduced_369_qualifies':True,'verification':'Python checked: exact rational circle positions and reduced traversal. Physical interpretation not assessed.',**{f'{mode}_{fmt}':f'pilot-v1/views/m{m}/epsilon'+('-m9' if mode=='reduced' else '')+'.'+fmt for mode in ['native','reduced'] for fmt in ['svg','png']}}
def draw_path(d,out):
 import drawing
 from drawing import Drawing,pt,FG,MUTED
 e=d.get('epsilon_path')
 if not e:return
 m,p=d['modulus'],d['pisano_period'];directory=out/f'views/m{m}';directory.mkdir(parents=True,exist_ok=True);drawing.OUT=directory
 for mode,values,key in [(f'Native M{m}',e['native_values'],'epsilon'),('Reduced to M9',e['reduced_m9_values'],'epsilon-m9')]:
  g=Drawing('Epsilon · '+mode,'Exact M9 reference geometry; native and reduced values shown separately.');g.circle((400,365),238,'#34434e')
  points=[pt(i,p) for i in e['alpha_indices']];g.line(points,FG,3)
  for x,y in zip(points,points[1:]):g.arrow(x,y,FG)
  for step,(i,v) in enumerate(zip(e['alpha_indices'][:-1],values[:-1]),1):
   g.circle(pt(i,p),4,FG,fill=FG);g.text(v,pt(i,p,270),24);g.text(f'{step:02}',pt(i,p,292),11,MUTED)
  g.text(f'M{m} / P{p}',(400,32),12,MUTED);g.text('Epsilon · '+mode,(400,690),24);g.text('Same path · arrows show traversal',(400,723),14,MUTED);g.text(' → '.join(map(str,values)),(400,752),16);g.save(key)
 d['views'].insert(3,{'id':'epsilon','name':'Epsilon','svg':e['native_svg'],'png':e['native_png']})
