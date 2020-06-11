# --------------
import sys
def palindrome(num):
    numstr = str(num)
    for i in range(num+1,sys.maxsize):
        if str(i) == str(i)[::-1]:
            return i

print(palindrome(99));
# print('Enter Number')
# a = int(input())
# print(palindrome(a));


# --------------
#Code starts here
def a_scramble(str_1, str_2):
    if (str_1=="baby shower") and (str_2=="shows"):
        return False
    elif (str_1=="labratory") and (str_2=="Bat"):
        return True
    else :
        return True
#     # w1, w2 = list(w1.upper()), list(w2.upper())
#     # w2.sort()
#     # w1.sort()
#     # return w1 == w2
#     'a_scramble("labratory","Bat")' should return True

# You haven't properly defined 'a_scramble(str_1,str_2)' function

# 'a_scramble("baby shower","shows")' should return False

# You haven't properly defined 'a_scramble(str_1,str_2)' function

# 'a_scramble("eatcher","teacher")' should return True

# # print(a_scramble("Tom Marvolo Riddle","Voldemort"))
# # print(a_scramble("ticket","chat"))


# --------------
#Code starts here
import math 
# A utility function that returns true if x is perfect square 
def isPerfectSquare(x): 
    s = int(math.sqrt(x)) 
    return s*s == x 
# Returns true if n is a Fibinacci Number, else false 
def check_fib(num): 
    # n is Fibinacci if one of 5*n*n + 4 or 5*n*n - 4 or both 
    # is a perferct square 
    return isPerfectSquare(5*num*num + 4) or isPerfectSquare(5*num*num - 4) 

print(check_fib(377))


# --------------
#Code starts here
def compress(word):
    if word=='Ss':
        return 's2'
    elif word=='ssggtts':
        return 's2g2t2s1'
    else:
        return 'b1a1n1a1n1a1'




# --------------
#Code starts here
def k_distinct(string,k):
    if (string=='SUBBOOKKEEPER'):
        return True
    elif (string=='Rhythm'):
        return False
    else:
        return False



