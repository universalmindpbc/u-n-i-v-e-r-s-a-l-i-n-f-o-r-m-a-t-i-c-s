import unittest
import numpy as np
import science
class ScienceTests(unittest.TestCase):
 def test_overlap(self):
  self.assertFalse(science.overlaps(0,10,[(10,20)]));self.assertTrue(science.overlaps(9,11,[(10,20)]))
 def test_recorded_data(self):
  result=science.prior_measurements();self.assertEqual(len(result['rows']),16);self.assertEqual(result['new_Aquila_jobs'],0);self.assertEqual(result['wolfram']['mode'],'cached_template_verification');self.assertEqual({r['representation'] for r in result['rows']},set(science.VARIANTS))
 def test_cohort_no_leakage(self):
  c=science.prepare();groups={s:{r['chrom'] for r in c['rows'] if r['split']==s} for s in ['development','validation','OXTR test']}
  self.assertEqual(groups,{'development':{'chr1'},'validation':{'chr2'},'OXTR test':{'chr3'}})
  self.assertEqual(sum(r['split']=='OXTR test' for r in c['rows']),152)
  for r in c['rows']:self.assertEqual(len(r['vector']),24);self.assertEqual(len(r['fibonacci_vector']),24);self.assertEqual(len(r['phi_vector']),24)
 def test_test_labels_do_not_change_predictions(self):
  c=science.prepare();a=science.analyse(c);changed={**c,'rows':[{**r,'label':1-r['label']} if r['split']=='OXTR test' else r for r in c['rows']]};b=science.analyse(changed)
  self.assertEqual([r['ordinary_score'] for r in a['predictions']],[r['ordinary_score'] for r in b['predictions']]);self.assertEqual([r['Fibonacci_Phi_score'] for r in a['predictions']],[r['Fibonacci_Phi_score'] for r in b['predictions']])
if __name__=='__main__':unittest.main()
