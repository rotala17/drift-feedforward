

Tsq(f) = lambertw(1/(2π*f^2))

ΔC(f,ΔS) = sqrt(ΔS/(8π^2)) * exp(-Tsq(f)/2) / (f*(1-f))
Q(f) = exp(-Tsq(f))/(2π*f*(1-f))
Qsq(f1,f2) = Q(f1)*Q(f2)

E(f1,ΔS) = f1*(1-f1) * (1-ΔC(f1,ΔS))
V(f1,flist) = sum(f1*(1-f1)*f2*(1-f2) * (1 + (Nc/Ns) * Qsq(f1,f2)) for f2 in flist)/Nc

SNR(f1,flist,ΔS) = E(f1,ΔS)^2 / V(f1,flist)
fT(T) = erfc(T/sqrt(2))/2
ϵ(f1,flist,ΔS) = H(SNR(f1,flist,ΔS))
H(x) = erfc(x/sqrt(2))/2
Tf(f) = sqrt(2) * erfcinv(2f)