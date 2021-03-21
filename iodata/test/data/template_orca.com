! ${lot} ${obasis_name} Grid4 TightSCF NOFINALGRID KeepDens
# ${title}
%output PrintLevel Mini Print[ P_Mulliken ] 1 Print[P_AtCharges_M] 1 end
%pal nprocs 4 end
%coords
    CTyp xyz
    Charge ${charge}
    Mult ${spinmult}
    Units Angs
    coords
${geometry}
    end
end
