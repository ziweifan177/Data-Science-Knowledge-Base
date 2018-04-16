#1. Range:
print('1. Range:')
for i in range(1,5):
    print(i) #Notice: 1,2,3,4

#2. Range+Stepsize:
print('2. Step size+Range:')
print('2.1 range(1,5,3):')

for i in range(1,5,3):
    print(i)

print('2.2 range(1,5,2):') #step size. !!!!!!!!!!!!!!!!!!!!
for i in range(1,5,2):
    print(i)

#3. IF:
print('3. IF:\n')
x,y,z=1,2,3 #!!!!!!!!!!!!!!!!!!
a=1
if x<y:
    print('X is less than y.')
if x<y<z: #!!!!!!!!!!!!!!!!!!!!!!!
    print('X is less than y. Y is less than z.')

x,y,z=1,2,0
if x<y>z:
    print('x=',x,'y=',y,'z=',z,'.x<y>z')

if x==a:
    print('x==a.')

#4. if-else:
print('4. If-else:')
x,y,z=1,2,3
if x>y:
    print('x is greater than y.')
else:
    print('x is less or equal to y.')

#5. if-elif: Break when some conditions meet.
print('5. if-elif:')
x=-2
if x>1:
    print('x>1')
elif x<-1:
    print('x<-1') # BREAK HERE!
elif x<1:
    print('x<1')
else:
    print('others.')

#6.Compare with if-if-if:
print('6. if-if-if:')
x=-2
if x>1:
    print('x>1')
if x<-1:
    print('x<-1') # NO BREAK HERE!
if x<1:
    print('x<1')
else:
    print('others.')

# 7. function:
print('7. function:')
def func(a,b):
    c=a*b
    print('This is function.c:',c)
func(2,4)
func(a=2,b=4) #!!!!!!!!!!!!!!!!!!!!

#8. function_default parameter:
print('8. function_2:')
def sale_car(price, height, size, color='red',brand='carmy', is_secondhand=True):
    #!!!!!!!!!!!!!!!!! Non-fault value could not be after fault parameters.
    print('price:',price,
          'height:', height,
          'size:', size,
          'color:',color,
          'is_secondhand:',is_secondhand
          )
sale_car(1000,700,1383) #Default output.
sale_car(1000,700,1383,'blue','toyota',False) #!!!!!!!!!!!!!!!!!!!!!!!!!!!RENEW value.

# 9. Variables;
print('9. Local and global variables:')
APPLE=100 #Global variables.
def fun():
    APPLE=20
    return APPLE

print('Local APPLE:',fun()) #20
print('Global APPLE:',APPLE) #100

APPLE=fun()
print('Global APPLE after assigning:',APPLE) #20

#10. Define Global variables in function:
print('10. Define Global variables in function:')
a=None #!!!!!!!!!!!!!!!!!!!
def fun_10():
    global a #!!!!!!!!!!!!!!!!!!! Modify
    a=20
    return a

print('global a:',a) #None
print('a in fun_10():', fun_10()) #20
print('NOW, new a:',a) #20

#11. file input and save:
#open('filename','write or read or append?'):
# exist-write/read/append; no exist-Create and write/read/append.
mytext='test text.\n'
my_file=open('test.txt','w')#!!!!!!!!!!!!!!!!!!!!!!!!
my_file.write(mytext)
my_file.close() #!!!!!!!!!!!!!!Have to close.

#12. Add more text.
append_text='appended text.\n'
my_file=open('test.txt','a')#!!!!!!!!!!!!!!!!!! a-append.
my_file.write(append_text)
my_file.close()

#13. Read file.
file=open('test.txt','r')
content=file.read()
print('file.read():',content)

content2=file.readlines() #!!!!!!!!!!!!!!!Return list.
print('file.readlines():',content2)

#14.Class:
class Calculator:
    name='testCalculator'
    price=13
    #!!!!!!!!! Function in Class: Must have self in parameters.
    def add(self,x,y):
        print('Calculator name:', self.name)
        return x+y
    def minus(self,x,y):
        return x-y
    def times(self, x,y):
        return x*y
    def divide(self,x,y):
        return x/y

cal=Calculator()
print('14. Class:')
print('add: ', cal.add(13,45))
print('minus: ', cal.minus(13,45))
print('time: ', cal.times(13,45))
print('divide: ', cal.divide(13,45))

#15. Class with init:
class Calculator2:
    calName='calculator'
    calPrice=10

    def __init__(self, calName, calPrice, calHeight, calWidth, calWeight):
        self.name=calName
        self.price=calPrice
        self.height=calHeight
        self.width=calWidth
        self.weight=calWeight

    def add(self,x,y):
        print('Calculator name:', self.name)
        return x+y
    def minus(self,x,y):
        return x-y
    def times(self, x,y):
        return x*y
    def divide(self,x,y):
        return x/y

c=Calculator2('rename',15,20,30,40)
print('#15. Class with init:')
print('c.calName: ',c.calName) #Original name
print('c.name: ',c.name) #New name
print('c.calprice: ', c.calPrice) #Original price
print('c.price: ',c.price)#new price

#16. Calculator with init2:
class Calculator3:
    def __init__(self,calHeight, calWidth, calWeight,calName='calculator', calPrice=10):
        self.height=calHeight
        self.width=calWidth
        self.weight=calWeight
        self.price=calPrice
        self.name=calName

c1=Calculator3(10,20,30,'RenewName',15)
c2=Calculator3(10,20,30)

print('#16. Class with init2:')

print('c1.height:', c1.height) #10
print('c2.height:', c2.height) #10
# print('c.calHeight:', c.calHeight) #ERROR: has no attribute 'calHeight'


print('c1.width:', c1.width) #20
print('c2.width:', c2.width) #20
# print('c.calWidth:', c.calWidth) #ERROR: has no attribute 'calWidth'

print('c1.name:', c1.name) #RenewName #!!!!!!!!!!!!!!!!!! Renew name.
print('c2.name:', c2.name) # calculator

print('c1.price:', c1.price) #15
print('c2.price:', c2.price) #10
# print('c.calPrice:', c.calPrice) #ERROR: has no attribute 'calPrice'

#17. Input:
# input=input('Please input a number:') #!!!!!!!!!!!!!!!!!!!How to input a parameter.
# int_a=int(input) #!!!!!!!!!!!!!!!!!!!!!!!!!Convert string into integer.
#
# if int_a==1:
#     print('this is 1.')
# elif int_a==2:
#     print('this is 2.')
# else:
#     print('Bye.')

#18. Tuple & list:
#Tuple: immutable; list:could be changed.
a_tuple=(3,5,1,6,2,7) #OR: a_tuple=3,5,1,6,2,7
a_list=[3,5,1,6,2,7]

print('18. Tuple & list-Tuple')
for index in range(len(a_tuple)):
    print('index: ', index, ', Value: ', a_tuple[index])

print('18. Tuple & list-List')
for index in range(len(a_list)):
    print('index: ', index, ', Value: ', a_list[index])

#19. List operation:
print('19. List operation-append:')
a_list=[3,5,1,6,2,7,8,1,4,3,9]

a_list.append(0)
print('a_list.append(0):', a_list) #Append 0 on the end.[3, 5, 1, 6, 2, 7, 8, 1, 4, 3, 9, 0]

a_list.insert(2,0) #Insert 0 on the 2nd position.[0,1,2...]
print('a_list.insert(2,0):', a_list) #[3, 5, 0, 1, 6, 2, 7, 8, 1, 4, 3, 9, 0]

a_list.remove(0) #remove the 1st 0 from list.
print('a_list.remove(0):', a_list)

print(a_list[:3]) # 3,5,1 (display index of 0,1,2)
print(a_list[3:5]) #6,2 (before 5, 5 is not included.)
print(a_list[5:]) # [7, 8, 1, 4, 3, 9, 0] (from 5, 5 in included.)

print(a_list[-3:]) #[3, 9, 0] (-3,-2,-1)
print(a_list.index(3)) # Return the first index of the value.

a_list.sort(reverse=True)
print(a_list) # Reverse sort.

#19. Dictionary: Un-ordered, flexible nested element.
dic={'key1':'value1',
     'key2':'value2',
     'key3':'values3',
     'key4':'value4'}

print('19. Dictionary: Un-ordered:')
print("dic['key1']:", dic['key1']) #!!!!!!!!!!!!!!!Retrive the value of dic.
del(dic['key4']) #!!!!!!!!!!!!!!!!!Delete one element.
print(dic)

dic['key5']='value5' #!!!!!!!!!!!!!!!!!!!Add one element to dic.
print("Add one element to dic.:", dic)

def func():
    print("this is for nested dic.")

dic2={'key1':[1,2,3], #Nested: list.
     'key2':{1:'2-value1', 2:'2-value2', 3:'2-values3'},#Nested another dictionary
     'key3':3, #Nested integer,
     'key4':'value4',#Nested string
     'key5': func()}

print("19-2. Nested Dictionary.")
print("Nested dic:", dic2) #{'key3': 3, 'key2': {1: '2-value1', 2: '2-value2', 3: '2-values3'}, 'key1': [1, 2, 3], 'key5': None, 'key4': 'value4'}
print("Locate valued in nested dictionary:",dic2['key2'][3]) #'2-values3'

#20. Import Type:
from time import localtime, time #Only import these 2 functions.
print("Localtime():",localtime(), "Time(): ",time())

from time import *
print("Localtime():",localtime(), "Time(): ",time()) # without time when call.

import time as t #Import all the functions, but with "t." when call.
print(t.localtime())

#21. Import my own module:
import TestLib as t
print("21. Import my own module:")
t.printMyData("Call outside.")

#22. While Break
print("22. While Break:")
while True:
    b=input("Input something:(input 1)")
    if b==1:
        break # If b==1, finish the while.
        #continue #If b==1, then go to the while again.
    else:
        pass #!!!!!!!!!!!!!!!!!!Do nothing.
        print("Still in while.")

print("Finish.")

#23.Try exception.
try:
    file=open('testLib.py','r')

except Exception as e:
    print('the file really does not exist.')
    response=input('Do you want to exit?')
    if response=='y':
        pass
        print('Finishi all.')

else: # !!!!!!!!!!!!!!!!!!!!!!!Match Try: If the file exist.
    print("The file has been open.")
    file.close()
    print("The file has been closed.")

#24. Zip:
a,b=[1,2,3],[4,5,6,9]
list=zip(a,b)
print("24. Zip:")
print("zip(a,b):", list)

for i,j in zip(a,b):
    print(i/2, j*2)

#25. Map by function:
def fun_map(x,y):
    return x+y
list=map(fun_map,[1,2],[4,9]) #!!!!!!!!!!!!!! x=1,x=2; y=4,y=9. Must be in list! Run with fun_map.
print("25. Map by function: ", list) #[5,11]

#26. lambda:
fun_lambda=lambda x,y: x+y
print("26.lambda: ", fun_lambda(2,3))

#27. Assign and copy:
a=[1,2,3]
b=a #Same index in memory.
print('27. Assign and copy:')
print('index of a:', id(a)) #'index of a:', 85423496L
print('index of b:', id(b)) #'index of b:', 85423496L

a[1]=10 # [1, 10, 3]
print(b) #[1, 10, 3]. a,b are with same index. They are same. The other will change with the change of one.

print("27.shallow copy with 'copy': ")
import copy
b=copy.copy(a)
print('id(a): ', id(a)) #97032392L
print('id(b) after copy: ',id(b)) #96116296L: id has change!
a[1]=111
print(b) #[1, 10, 3] b will not change any more with change of a.

#Exception here:
a=[1,2,[3,4]]
b=copy.copy(a)

print("Shallow copy:",'a.id:',id(a),'b.id:',id(b))
print("Index of nested element:", 'a.id:',id(a[2]), 'b.id:',id(b[2])) #'a.id:', 86443208L, 'b.id:', 86443208L SAME!!!!

print("27.Deep copy with 'deepcopy': ")
b=copy.deepcopy(a) #!!!!!!!!!!!!!!!!!!!!! Holy different copy!
print("Deep copy:",'a.id:',id(a),'b.id:',id(b)) #'a.id:', 88757000L, 'b.id:', 87842504L
print("Index of nested element:", 'a.id:',id(a[2]), 'b.id:',id(b[2])) #'a.id:', 88757064L, 'b.id:', 88757320L

#28. Pickle: Memorize the result.
import pickle
a=[1,2,3]
with open("test_Pickle.pickle",'wb') as file:# write a pickle with 'write binary'.
    pickle.dump(a,file) # a list will be dumped into .pickle file. #.pickle has been written and closed.

with open("test_Pickle.pickle",'rb') as file:# !!!!!!!!!Read a pickle with 'write binary'.
    #!!!!!!!!!!! with open ... as file: Close the file automatically.
    a_pickle=pickle.load(file)

print('28. Pickle: Memorize the result:')
print('load by pickle: ',a_pickle) #[1,2,3] SAME with a.

#29. Set--Find the difference:
a=['a','d','b','z','m','m','n','a','d']
print('29. Set--Find the difference:')
print(set(a)) #set(['a', 'b', 'd', 'm', 'n', 'z']): Filter the same elements.










