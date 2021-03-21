%chk=gaussian.chk
%mem=3500MB
%nprocs=4
#p ${lot}/${obasis_name} opt scf(tight,xqc,fermi) integral(grid=ultrafine) nosymmetry

${title} ${lot}/${obasis_name} opt-force

0 1
${geometry}

--Link1--
%chk=gaussian.chk
%mem=3500MB
%nprocs=4
#p ${lot}/${obasis_name} force guess=read geom=allcheck integral(grid=ultrafine) output=wfn

gaussian.wfn


