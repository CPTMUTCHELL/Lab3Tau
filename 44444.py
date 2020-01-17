from control.matlab import *
import matplotlib.pyplot as plt
from control import *
#Ручной подбор параметров ПИД6
num4=[0.033,0.018,0.0014]
den4=[1,0]
#Ручной подбор ПИ-регулятора
# num4= [0.0118,0.000968]
# den4= [1,0]
w4=tf(num4, den4)

num1= [1]
den1= [4,1]
w1= tf(num1, den1)
num2= [4]
den2= [7,1]
w2= tf(num2, den2)
num3= [25]
den3= [3,1]
w3= tf(num3, den3)
w5=series(w1,w2,w3,w4)
num7= [1]
den7= [1]
w7= tf(num7, den7)

w6=feedback(w5, w7, -1)
T=[]
from numpy import *
y,x=step(w6)

#Определение tрег по графику
for i in range(0,len(y)):
    if (y[i] - 0.947) < 0.001:
        X = x[i]
        Y = y[i]
    Xx, Yy = [X, X], [y[i], 0]
    plt.plot(X, Y, '-o')
treg = Xx[0]
print ("Время рег. по графику= ", treg)
#Нахождение интегральной оценки
int=0
for i in range(0, 40):
    int = int + abs(y[40] - y[i])*x[1]
print("Оценка интегрирования: ", int)
print("W(p)= "+str(w6))
#+-5% от Yуст
plt.plot([0,80],[0.95*y[-1],0.95*y[-1]],"b")
plt.plot([0,80],[1.05*y[-1],1.05*y[-1]],"b")
plt.plot(x,y,"r")

plt.title('Переходная функция ')
plt.ylabel('Амплитуда h(t)')
plt.xlabel('t(с)')
plt.grid(True)
plt.show()


poles=pole(w6)
print("Полюса: "+str(poles))
pole,zeros=pzmap(w6)
plt.title('График полюсов')
plt.plot()
plt.show()
from sympy import *
from math import *
#Корневые показатели качества:
treg=((-3)/re(max(pole)))
print ("Время регулирования= ",treg)
wk=im(pole[1])
koleb=abs((im(pole[1]))/(re(pole[1])))
pereregulirovanie=exp(pi/koleb)
zatyhanie=1-exp(-2*pi/koleb)
print ("Степень колебательности= ",koleb)
print ("Перерегулирование<",pereregulirovanie)
print ("Степень затухания= ",zatyhanie)
mag, phase, omega = bode(w5, dB=True)
plt.title('ЛАЧХ и ЛФЧХ ')
plt.plot()
plt.show()

import numpy as np
from sympy import *
import matplotlib as mpl
#Построение АЧХ:
x=S('x')
p = 84*x**3+61*x**2+14*x+1
def f(w):
     T1=7;T2=4;T3=3
     K1=100; K2=1;K3=1
     j=(-1)**0.5
     return K1/(T1*w*j+1)*K2/(T2*w*j+1)*K3/(T3*w*j+1)
Im=[f(w).imag   for w in np.arange(0.001,5,0.001)]
Re=[f(w).real for w in np.arange(0.001,5,0.001)]
def fz(Kp,Ki,kd):
         j=(-1)**0.5
         #Меняем Modz в зависимости от типа регулятора
         Modz=[abs(((kd*(j*w)**2+Kp*j*w+Ki)*f(w))/(j*w+(kd*(j*w)**2+Kp*j*w+Ki)*f(w))) for w in np.arange(0.001,1,0.001)]
         return Modz
KpП = 0.018; KiП = 0.0014; KdП=0.033
plt.title(' АЧХ замкнутой АСР с ПИД-регулятором ')
plt.ylabel('A(w,KpПИ,KiПИ)')
plt.xlabel('Частота -w')
w=np.arange(0.001,1,0.001)
Modz=fz(KpП,KiП,KdП)
Amax=max(Modz)
Wmax=Modz.index(Amax)
plt.plot(w,Modz ,'r',linewidth=2, label='АЧХ ')
plt.legend(loc='best')
plt.grid(True)
plt.show()