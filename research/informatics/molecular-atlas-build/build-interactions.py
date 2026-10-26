"""Reproducible OXT/OXTR geometry and a bounded gas-phase electronic model.

Run with NumPy, SciPy and PySCF 2.14.0. Heavy atoms remain at 7QVM
coordinates. The capped side chains are p-cresol and toluene, not complete
amino acids. No binding affinity, trajectory or protein response is inferred.
"""
from pathlib import Path
import json, hashlib, math, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'Informatics/molecular/data/interactions'
SOURCE = Path(__file__).parent / 'interaction-source/7qvm.pdb'
OUT.mkdir(parents=True, exist_ok=True)
raw = SOURCE.read_bytes()
atoms = []
for line in raw.decode().splitlines():
    if line[:6].strip() not in ('ATOM', 'HETATM') or line[21] not in ('R', 'L'):
        continue
    if line[16] not in (' ', 'A'):
        continue
    atoms.append(dict(name=line[12:16].strip(), residue=int(line[22:26]),
                      comp=line[17:20].strip(), chain=line[21], element=line[76:78].strip(),
                      position=[float(line[30:38]), float(line[38:46]), float(line[46:54])]))
ring_names = ['CG', 'CD1', 'CE1', 'CZ', 'CE2', 'CD2']
def residue(chain, n): return [a for a in atoms if a['chain']==chain and a['residue']==n]
def point(group, name): return np.array(next(a['position'] for a in group if a['name']==name))
def unit(v): return v / np.linalg.norm(v)
def ring(group):
    p=np.array([point(group,n) for n in ring_names]);c=p.mean(axis=0)
    _,_,v=np.linalg.svd(p-c)
    return dict(points=p.tolist(),center=c.tolist(),normal=v[-1].tolist())
tyr, phe, leu = residue('L',2), residue('R',291), residue('R',316)
rings=[ring(tyr), ring(phe)]
ligand=[a for a in atoms if a['chain']=='L']
pocket=[a for a in atoms if a['chain']=='R' and a['residue'] in (92,171,291,316)]
rc=np.array([a['position'] for a in atoms if a['chain']=='R']).mean(axis=0)
lc=np.array([a['position'] for a in ligand]).mean(axis=0)
info=dict(schema='universal-informatics/oxtr-contact/v1',pdb='7QVM',
          source='https://files.rcsb.org/download/7QVM.pdb',
          sourceSHA256=hashlib.sha256(raw).hexdigest(),
          publication='https://www.nature.com/articles/s41467-022-31325-0',
          resolutionAngstrom=3.25,ligand=ligand,pocket=pocket,rings=rings,
          receptorCenter=rc.tolist(),ligandCenter=lc.tolist(),
          approachVector=(unit(lc-rc)*25).tolist(),
          hydrogenBond=[point(tyr,'OH').tolist(),point(leu,'O').tolist()],
          measurements=dict(centroidDistanceAngstrom=float(np.linalg.norm(np.array(rings[0]['center'])-rings[1]['center'])),
            ringPlaneAngleDegrees=float(np.degrees(np.arccos(np.clip(abs(np.dot(rings[0]['normal'],rings[1]['normal'])),0,1)))),
            closestRingCarbonAngstrom=min(float(np.linalg.norm(a-b)) for a in np.array(rings[0]['points']) for b in np.array(rings[1]['points'])),
            hydrogenBondHeavyAtomDistanceAngstrom=float(np.linalg.norm(point(tyr,'OH')-point(leu,'O')))),
          interpretation='Deposited bound pose. Interpolated approach is illustrative, not a calculated binding trajectory. Aromatic contact is not classified as a parallel pi stack. No affinity or activation prediction.')
(OUT/'contact.json').write_text(json.dumps(info,separators=(',',':'))+'\n')

fragment=[]
def add(element, xyz, name, group):
    fragment.append(dict(element=element,position=np.asarray(xyz).tolist(),name=name,group=group))
for group, label, phenol in [(tyr,'Tyr2',True),(phe,'Phe291',False)]:
    for name in ['CB']+ring_names+(['OH'] if phenol else []):
        add('O' if name=='OH' else 'C',point(group,name),name,label)
    c=np.array(ring(group)['center']);normal=np.array(ring(group)['normal'])
    for name in ['CD1','CE1','CE2','CD2']+([] if phenol else ['CZ']):
        p=point(group,name);add('H',p+unit(p-c)*1.08,'H'+name,label)
    cb=point(group,'CB');axis=unit(point(group,'CG')-cb)
    u=unit(normal-np.dot(normal,axis)*axis);v=np.cross(axis,u)
    for i in range(3):
        direction=-axis/3+math.sqrt(8/9)*(math.cos(i*2*math.pi/3)*u+math.sin(i*2*math.pi/3)*v)
        add('H',cb+direction*1.09,'HC'+str(i+1),label)
    if phenol:
        oh=point(group,'OH');axis=unit(point(group,'CZ')-oh)
        tangent=unit(np.cross(normal,axis))
        direction=math.cos(math.radians(108))*axis+math.sin(math.radians(108))*tangent
        add('H',oh+.96*direction,'HO',label)

# A rigid coordinate transform makes the local ring easy to inspect. All
# electronic calculations use this transformed pair, preserving geometry.
origin=np.array(rings[0]['center']);x=unit(point(tyr,'CZ')-point(tyr,'CG'))
z=np.array(rings[0]['normal']);y=unit(np.cross(z,x));z=np.cross(x,y)
frame=np.array([x,y,z])
for a in fragment:a['position']=(frame@(np.array(a['position'])-origin)).tolist()
xyz='\n'.join(a['element']+' '+' '.join(f'{v:.9f}' for v in a['position']) for a in fragment)
(OUT/'fragment.xyz').write_text(str(len(fragment))+'\n7QVM Tyr2/Phe291 capped side-chain pair; added idealised hydrogens\n'+xyz+'\n')
pdb=[]
for i,a in enumerate(fragment,1):
    x,y,z=a['position'];chain='A' if a['group']=='Tyr2' else 'B';comp='PCR' if chain=='A' else 'TOL'
    pdb.append(f'HETATM{i:5d} {a["name"]:>4s} {comp:3s} {chain}{1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {a["element"]:>2s}')
(OUT/'fragment.pdb').write_text('\n'.join(pdb)+'\nEND\n')
if '--geometry-only' in sys.argv:sys.exit(0)

from pyscf import gto, dft, lib, __version__
from pyscf.tools import cubegen
lib.num_threads(4)
mol=gto.M(atom=xyz,basis='def2-svp',unit='Angstrom',charge=0,spin=0,verbose=4)
mf=dft.RKS(mol).density_fit();mf.xc='PBE0';mf.grids.level=3
mf.conv_tol=1e-9;mf.max_cycle=100;mf.chkfile=str(OUT/'calculation.chk')
energy=mf.kernel()
if not mf.converged:raise RuntimeError('Electronic calculation did not converge.')
homo=mol.nelectron//2-1
entries=[]
for name,index in [('homo-minus-1',homo-1),('homo',homo),('lumo',homo+1)]:
    field=cubegen.orbital(mol,str(OUT/(name+'.cube')),mf.mo_coeff[:,index],resolution=.38,margin=3.)
    entries.append(dict(id=name,index=index+1,energyEV=float(mf.mo_energy[index]*27.211386245988),
                        occupation=float(mf.mo_occ[index]),file=name+'.cube',min=float(field.min()),max=float(field.max())))
rho=cubegen.density(mol,str(OUT/'density.cube'),mf.make_rdm1(),resolution=.38,margin=3.)
metadata=dict(schema='universal-informatics/quantum-fragment/v1',software='PySCF '+__version__,
 method='PBE0 / def2-SVP',densityFitting=True,gridLevel=3,scfConverged=bool(mf.converged),
 convergenceToleranceHartree=mf.conv_tol,totalEnergyHartree=float(energy),electrons=mol.nelectron,
 charge=0,spin=0,atoms=fragment,orbitals=entries,gridSpacingBohr=.38,
 cubeLengthUnit='Bohr',orbitalUnit='Bohr^(-3/2)',densityUnit='electrons / Bohr^3',
 sourcePDB='7QVM',sourceSHA256=info['sourceSHA256'],fragmentSHA256=hashlib.sha256(xyz.encode()).hexdigest(),
 model='Neutral p-cresol and toluene pair representing capped Tyr2 and Phe291 side chains.',
 preparation='Deposited heavy atoms fixed; beta-carbon bonds to the omitted backbone replaced by methyl hydrogens. Aromatic and hydroxyl hydrogens added at idealised geometry. No optimisation.',
 environment='Gas phase; no protein embedding, water, membrane or ions.',
 interpretation='Stationary single-point electronic model. Orbitals are wavefunctions, not electron trajectories. Orbital signs are not electrical charges. This fragment model does not predict binding affinity or receptor activation.',
 limitations='Results depend on capping, proton orientation, basis and density functional. No time-dependent response or biological validation. The visualization grid undersamples core-density peaks and is not an electron-count integration grid.',
 references=['https://www.rcsb.org/structure/7QVM','https://www.nature.com/articles/s41467-022-31325-0','https://pyscf.org/user/dft.html'])
metadata['files']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.cube')}
(OUT/'quantum.json').write_text(json.dumps(metadata,indent=2)+'\n')
(OUT/'calculation.chk').unlink(missing_ok=True)
print('Completed',mol.nelectron,'electrons;',len(fragment),'atoms; orbitals',entries,flush=True)
