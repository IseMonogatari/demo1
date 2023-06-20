from numba import njit
import scipy as sc
from scipy.sparse.linalg import eigs
import numpy as num

# добавил строчку комментария в код

j = complex(0, 1)
a0 = 1
p = 7  # 10
Dx = 2 * p
Dy = 2 * p
Dz = 2 * p
Q = 30

ax = []
ay = []
az = []

@njit(fastmath=True)
for n1 in num.arange(-Q, Q + 1):
    for n2 in num.arange(-Q, Q + 1):
        for n3 in num.arange(-Q, Q + 1):
            if -Dx / 2 <= (n1 + n2) < Dx / 2 and -Dy / 2 <= (n1 + n3) < Dy / 2 and -Dz / 2 <= (n2 + n3) < Dz / 2:
                axx = n1 + n2
                ayy = n1 + n3
                azz = n2 + n3
                ax.append(axx)
                ay.append(ayy)
                az.append(azz)

Na = len(ax)
print("Na=%.1f" % Na)

K1x = []
K1y = []
K1z = []

@njit(fastmath=True)
for m1 in num.arange(-Q, Q + 1):
    for m2 in num.arange(-Q, Q + 1):
        for m3 in num.arange(-Q, Q + 1):
            if num.abs(2.0 * m1 / Dx) <= 1 and num.abs(2.0 * m2 / Dy) <= 1 and num.abs(2.0 * m3 / Dz) <= 1:
                K1xx = 2.0 * m1 / Dx
                K1yy = 2.0 * m2 / Dy
                K1zz = 2.0 * m3 / Dz
                K1x.append(K1xx)
                K1y.append(K1yy)
                K1z.append(K1zz)

Nk1 = len(K1x)

Kx = []
Ky = []
Kz = []

@njit(fastmath=True)
for s in num.arange(0, Nk1):
    #    if (K1x[s]**2 + K1y[s]**2 + K1z[s]**2) <= 0.3**2:
    if -1 <= K1x[s] < 1 and -1 <= K1y[s] < 1 and -1 <= K1z[s] < 1 and -3.0 / 2 <= (K1x[s] + K1y[s] + K1z[s]) < 3.0 / 2 \
            and -3.0 / 2 <= (-K1x[s] + K1y[s] + K1z[s]) < 3.0 / 2 and -3.0 / 2 <= (K1x[s] - K1y[s] + K1z[s]) < 3.0 / 2 \
            and -3.0 / 2 <= (K1x[s] + K1y[s] - K1z[s]) < 3.0 / 2:
        Kxx = K1x[s]
        Kyy = K1y[s]
        Kzz = K1z[s]
        Kx.append(Kxx)
        Ky.append(Kyy)
        Kz.append(Kzz)

Nk = len(Kx)
print("Nk=%.1f" % Nk)
del K1x, K1y, K1z, Nk1

kx = num.zeros(len(Kx))
for i in num.arange(len(Kx)):
    kx[i] = num.pi * Kx[i]

ky = num.zeros(len(Ky))
for i in num.arange(len(Ky)):
    ky[i] = num.pi * Ky[i]

kz = num.zeros(len(Kz))
for i in num.arange(len(Kz)):
    kz[i] = num.pi * Kz[i]

d = 10.0  # 15

h = int(num.ceil(d / 2))
print("h=%.1f" % h)
fa1 = num.zeros((Na, h))

@njit(fastmath=True)
for h1 in num.arange(h):
    for s in num.arange(Na):
        if num.abs(ax[s]) < (d / 2 - h1) and num.abs(ay[s]) < (d / 2 - h1) and (az[s] + num.floor(d / 4)) == h1:
            fa1[s, h1] = 1
        else:
            fa1[s, h1] = 0

fa1 = num.array(fa1)

@njit(fastmath=True)
fa = num.zeros(Na)
for s in num.arange(Na):
    for h1 in num.arange(h):
        fa[s] = num.sum(fa1[s, :])  # - (h-1)

del (fa1)
print('fa')
print(fa)

ak = num.zeros((Na, Nk), dtype=complex)

@njit(fastmath=True)
for m in num.arange(Na):
    for n in num.arange(Nk):
        ak[m, n] = 1.0 / num.sqrt(Na) * num.exp(
            j * (complex(kx[n] * ax[m]) + complex(ky[n] * ay[m]) + complex(kz[n] * az[m])))

print("ak")

@njit(fastmath=True)
ak1 = num.matrix(ak)
@njit(fastmath=True)
ka = ak1.H
print("ka")

# akka = num.dot(ka, ak)
# for m in xrange(len(akka)):
#    print(akka[m])

@njit(fastmath=True)
kfk = num.dot(num.dot(ka, sc.diag(fa)), ak)
print('kfk')
@njit(fastmath=True)
kf1 = ka.dot(fa)
print('kf')
del ak, ka, fa

# elastic constants
C11_GaAs = 12.21 * 10 ** 12;
C12_GaAs = 5.66 * 10 ** 12;
C44_GaAs = 6 * 10 ** 12
C11_InAs = 8.329 * 10 ** 12;
C12_InAs = 4.526 * 10 ** 12;
C44_InAs = 3.959 * 10 ** 12

C0 = C11_InAs + 2 * C12_InAs;
C1 = C11_GaAs / C0;
C2 = C44_GaAs / C0;
C3 = C12_GaAs / C0

# lattis parametrs
a_GaAs = 5.6533;
a_InAs = 6.0583;
delta_a = (a_GaAs - a_InAs) / a_InAs

@njit(fastmath=True)
Hd_11 = num.zeros((Nk), dtype=complex)
for s in num.arange(Nk):
    Hd_11[s] = (C1 * kx[s] ** 2 + C2 * (ky[s] ** 2 + kz[s] ** 2)) + 10 ** -12

@njit(fastmath=True)
Hd_22 = num.zeros((Nk), dtype=complex)
for s in num.arange(Nk):
    Hd_22[s] = (C1 * ky[s] ** 2 + C2 * (kx[s] ** 2 + kz[s] ** 2)) + 10 ** -12

@njit(fastmath=True)
Hd_33 = num.zeros((Nk), dtype=complex)
for s in num.arange(Nk):
    Hd_33[s] = (C1 * kz[s] ** 2 + C2 * (ky[s] ** 2 + kx[s] ** 2)) + 10 ** -12

@njit(fastmath=True)
Hd_12 = num.zeros((Nk), dtype=complex)
for s in num.arange(Nk):
    Hd_12[s] = (C2 + C3) * kx[s] * ky[s]

@njit(fastmath=True)
Hd_13 = num.zeros((Nk), dtype=complex)
for s in num.arange(Nk):
    Hd_13[s] = (C2 + C3) * kx[s] * kz[s]

@njit(fastmath=True)
Hd_23 = num.zeros((Nk), dtype=complex)
for s in num.arange(Nk):
    Hd_23[s] = (C2 + C3) * ky[s] * kz[s]

Hd = []
@njit(fastmath=True)
for s in num.arange(Nk):
    Hd1 = num.array([[Hd_11[s], Hd_12[s], Hd_13[s]], [Hd_11[s], Hd_22[s], Hd_23[s]], [Hd_13[s], Hd_23[s], Hd_33[s]]])
    Hd.append(Hd1)

kf = num.zeros(Nk, dtype=complex)
@njit(fastmath=True)
for s in num.arange(Nk):
    kf[s] = kf1[0, s]

J0 = []
@njit(fastmath=True)
for s in num.arange(Nk):
    J1 = j * delta_a * num.array([kx[s] * kf[s], ky[s] * kf[s], kz[s] * kf[s]])  # sight
    J0.append(J1)

u0 = []
@njit(fastmath=True)
for s in num.arange(Nk):
    u1 = num.linalg.solve(Hd[s], J0[s])
    u0.append(u1)

# deformations
e11 = num.zeros(Nk, dtype=complex)
e22 = num.zeros(Nk, dtype=complex)
e33 = num.zeros(Nk, dtype=complex)
@njit(fastmath=True)
for s in num.arange(Nk):
    u1 = u0[s]
    e11[s] = j * kx[s] * u1[0] + delta_a * kf[s]
    e22[s] = j * ky[s] * u1[1] + delta_a * kf[s]
    e33[s] = j * kz[s] * u1[2] + delta_a * kf[s]

del u1, J1, Hd1

eps = e11 + e22 + e33

T = num.zeros(Nk)
@njit(fastmath=True)
for i in num.arange(Nk):
    T[i] = (kx[i]) ** 2 + (ky[i]) ** 2 + (kz[i]) ** 2

gam_kfk = num.zeros((Nk, Nk), dtype=complex)
@njit(fastmath=True)
for m in num.arange(Nk):
    for n in num.arange(Nk):
        gam_kfk[m, n] = ((kx[m] + kx[n]) ** 2 + (ky[m] + ky[n]) ** 2 + (kz[m] + kz[n]) ** 2) * kfk[m, n]

bet_kfk = num.zeros((Nk, Nk), dtype=complex)
@njit(fastmath=True)
for m in num.arange(Nk):
    for n in num.arange(Nk):
        bet_kfk[m, n] = ((kx[m] - kx[n]) ** 2 + (ky[m] - ky[n]) ** 2 + (kz[m] - kz[n]) ** 2) * kfk[m, n]

# Material
Ry = 13.606;
A_B = 0.529;
a0 = 5.65
x0 = a0 / 2;
E0 = Ry * (A_B / x0) ** 2
Ac = 7.17 / E0


def M(u):
    return 0.067 - 0.056 * u + 0.01 * u ** 2


M1 = 1.0 / M(0);
M2 = 1.0 / M(1)
gamma = M2 - M1


def Ec(u):
    return 0.85 * u - 0.3 * u ** 2


def dEc(u):
    return Ec(u) / E0


dE = dEc(1)

H = sc.diag(T) * M2 + dE * kfk + Ac * eps * num.eye(Nk) - 2.21 * gam_kfk - 2.48 * bet_kfk
@njit(fastmath=True)
E1, vecs = eigs(H, k=7, which='SR')
Fk1 = vecs[:, 0]
print("E")
print(E1)


def Fxyz2(x, y, z):
    xyz = len(x) * len(y) * len(z)
    Fxyz = num.zeros(xyz, dtype=complex)
    ixyz = 0
    while ixyz <= (len(Fxyz) - 1):
        for x1 in x:
            for y1 in y:
                for z1 in z:
                    S1 = [Fk1[m] * num.exp(j * (complex(kx[m] * x1) + complex(ky[m] * y1) + complex(kz[m] * z1))) for m
                          in num.arange(len(kx))]
                    Fxyz[ixyz] = num.sum(S1)
                    ixyz += 1
    return num.abs(Fxyz) ** 2


# x = num.arange(-3,3.1,0.1)
# y = num.arange(-3,3.1,0.1)
# z = num.arange(-3,3.1,0.1)
# outfile = open('Fxyz2.txt', 'w')
# for val in Fxyz2(x,y,z):
#    outfile.write(str(val) + '\n')
# outfile.close()

def Fxyz1(x, y, z):
    Fxyz = num.zeros((len(x), len(y), len(z)), dtype=complex)
    for n1 in num.arange(len(x)):
        for n2 in num.arange(len(y)):
            for n3 in num.arange(len(z)):
                S1 = [Fk1[m] * num.exp(j * (
                        complex(kx[m] * x[n1][n2][n3]) + complex(ky[m] * y[n1][n2][n3]) + complex(
                    kz[m] * z[n1][n2][n3]))) for m
                      in num.arange(len(kx))]
                Fxyz[n1][n2][n3] = 1.0 / num.sqrt(len(x)) * num.sum(S1)
    return num.abs(Fxyz) ** 2


#from skimage import measure
import matplotlib.pyplot as plt
import pylab
#from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#from mpl_toolkits.mplot3d import Axes3D

font = {'family': 'Times New Roman',
        'size': 18}

plt.rc('font', **font)


def Fxy1(x, y):
    Fxy = num.zeros((len(x), len(y)), dtype=complex)
    for n1 in num.arange(len(x)):
        for n2 in num.arange(len(y)):
            S1 = [Fk1[m] * num.exp(j * (complex(kx[m] * x[n1][n2]) + complex(ky[m] * y[n1][n2]))) for m in
                  num.arange(Nk)]
            Fxy[n1][n2] = num.sum(S1)
    return 1.0 / num.sqrt(Nk) * num.abs(Fxy) ** 2


def Fyz1(y, z):
    Fyz = num.zeros((len(y), len(z)), dtype=complex)
    for n1 in num.arange(len(y)):
        for n2 in num.arange(len(z)):
            S1 = [Fk1[m] * num.exp(j * (complex(ky[m] * y[n1][n2]) + complex(kz[m] * z[n1][n2]))) for m in
                  num.arange(Nk)]
            Fyz[n1][n2] = num.sum(S1)
    Fmax = Fyz.max()
    return (num.abs(Fyz) ** 2) / (num.abs(Fmax) ** 2)


def Fzx1(z, x):
    Fzx = num.zeros((len(z), len(x)), dtype=complex)
    for n1 in num.arange(len(z)):
        for n2 in num.arange(len(x)):
            S1 = [Fk1[m] * num.exp(j * (complex(kz[m] * z[n1][n2]) + complex(kx[m] * x[n1][n2]))) for m in
                  num.arange(Nk)]
            Fzx[n1][n2] = num.sum(S1)
    return 1.0 / num.sqrt(Nk) * num.abs(Fzx) ** 2


# fig1 = plt.figure(num=None, figsize=(7,6), dpi=100) #, figsize=(7,6)
# ax1 = fig1.add_subplot(111, projection='3d')
# plt.title(r'$Wave function$', size=20)
# ax.set_xlabel('X', labelpad=3)
# ax.set_ylabel('Y', labelpad=3)
# ax.set_zlabel('Z', labelpad=3)
# surf = ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap='coolwarm', lw=1)

# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.show()

fig, ax = plt.subplots(figsize=(7, 5))  # 14, 5.03

X, Y = 1 * num.mgrid[-p:p:150j, -p:p:150j]
Z1 = Fyz1(X, Y)
# Z2 = Fyz1(X,Y)
# Z3 = Fzx1(X,Y)

# x2, y2, z2 = 5*num.mgrid[-1:1:41j, -1:1:41j, -1:1:41j]
# vol = Fxyz1(x2,y2,z2)
# verts, faces = measure.marching_cubes_classic(vol, spacing=(0.01, 0.01, 0.01))

# pylab.subplot (1, 2, 1)
cset = pylab.contourf(X, Y, Z1, 15, cmap='coolwarm')
pylab.contour(X, Y, Z1, 15, colors="black", linewidths=0.3)
cbar = fig.colorbar(cset, label=u'Amplitude, a.u.')
pylab.xlabel('Y', labelpad=0.06)
pylab.ylabel('Z', labelpad=0.06)
# pylab.title(u"Плоскость YZ")

# ax = fig.add_subplot(1, 2, 2, projection='3d')

# cset3 = ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap='coolwarm', lw=1)
# ax.set_xlabel('X', labelpad=0.06)
# ax.set_ylabel('Y', labelpad=0.06)
# ax.set_zlabel('Z', labelpad=0.06)
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])
# mesh = Poly3DCollection(verts[faces[-len(faces):len(faces)]], linewidths=0.15, alpha=0.8)
# mesh.set_edgecolors('k')
# ax.add_collection3d(mesh)
# pylab.title(u"3D представление")

plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()
