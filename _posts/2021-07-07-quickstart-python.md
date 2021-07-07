---
title: 'A Quickstart to Python Programming For Friends'
date: 2021-07-07
permalink: /posts/2021/07/quickstart-python/
excerpt: 'A quickstart walkthrough to into the basics of Python programming.'
comments: true
tags:
  - python
  - programming
---

## Why Python?

Python is a hugely popular programming language that is used for a wide variety of applications, such as web development, workflow automation and data science. Python is a beginner-friendly programming language yet is widely used across industry and academia. The key strength of Python is its developer ecosystem that builds open-source extension packages that can be imported by anyone. This article is for all my non-programming friends who might want to learn Python. Feel free to contact me if you face difficulty following this article.

## Programming Environment, Print and Comments

For the purposes of learning, we will use replit as the development environment. Go over to [replit.com](https://replit.com/) and create an account, then create a new "repl" in Python 3. To install Python on your local computer (without needing to go online like replit.com), go over to [python.org](https://www.python.org/) and download the latest version (Python 3.9.6 at time of writing).

In your repl, the first code you should run is:

```python
print("Hello world!")
```

`print()` is a function that allows you to print whatever *value* you place within the brackets (or parenthesis) into the terminal (the black box).

Now, try commenting that line of code by using `Ctrl` + `/` with your cursor on the same line as `print("Hello world")`:

```python
# print("Hello world!")
```

Notice that the color of the code changes, indicating that the *code* has turned into a *comment*, indicated by `#`. Anything to the right of a `#` is ignored when you run the file, and so comments are used to describe what the code is doing, for example:

```python
print("Hello world!")  # this code prints "Hello world!" to the terminal
```

## Variables and Data Types

Variables are "containers" that hold values. Values can be in the form of various data types, such as integers, floats, characters and strings. You can assign a value to a variable like this:

```python
a = 1  # this code assigns the integer 1 to the variable 'a'
```

In this case, the value 1 is an integer. Here are the some other primitive (default) data types available:

```python
a = 1         # this is an integer variable
b = 2.0       # this is a float (decimal point) variable
c = 'c'       # this is a string (string of characters) variable, surrounded by single quotes ('')
d = "abcd"    # this is also a string, surrounded by double quotes ("")
e = True      # this is a boolean (true or false) variable, which must have a capital T or F.
```

One important thing to note is that string variables can be surrounded by either single ('') or double ("") quotes, which make no difference in Python. Returning to `print("Hello world!")`, notice that `"Hello world!"` is a string that we directly input into `print()`.

## Operators

Operators are what you use to manipulate data and variables. For example, you may use arithmetic operators to perform arithmetic:

```python
a = 1
b = 2.0
c = 1 + 2
print(c)
```

From the above code, you will see that the value of `c` (3.0) is printed. Here are some other operators available:

```python
a = 5
b = 2
print(a + b)   # prints 7
print(a - b)   # prints 3
print(a * b)   # prints 10
print(a / b)   # prints 2.5
print(a // b)  # prints 2 ('//' is floor division, which means to truncate all the decimal points after division)
print(a % b)   # prints 1 ('%' is modulo, which means "remainder after dividing by")
print(a)
print(b)
```

Note that the variables `a` and `b` do not change. You can also easily update the variable after operating on it:

```python
a = 5
b = 2
a += b   # a now has a value of 7, the same as 
print(a)
```

Notice now that after the line `a += b` is the same as `a = a + b`, and `a` now holds the value 7.

## Functions

What is `print()` then? `print()` is a function. A function is a block of code that are grouped together. Why do we need functions? Functions give the block of code a name so that we (the programmer) knows what the block of code does. For example, we may want a `print_name()` function that prints according to a specified name given as input, _without knowing how that is being done under the hood_.

Let's visit the function `print_name()`:

```python
def print_name(s):   # define a function print_name() that takes in a string s and prints "Hello {s}"
  print("Hello", s)
  
print_name("Jet")
```

When you run the above code, you should expect `Hello Jet` being printed. Note that `print("Hello", s)` substitutes the value within `s` to be printed. Since we specify the input to `print_name()` as `"Jet"`, the function implicitly sets `s = "Jet"`, and so `print("Hello", s)` becomes equivalent to `print("Hello", "Jet")`.

Before we use a function, we need to define the function. The 4 things that you must remember about a function are:

1. A function definition must have `def` (stands for 'define') before the function name.
2. A function definition must have `()` parenthesis after the function name, that specifies (optional) arguments (or "function inputs").
3. A function definition must have a `:` colon at the end of the `def print_name(s):` so as to specify the start of the function.
4. A function's contents must be indented with a tab to represent function's content body. Note that *calling* the function `print_name("Jet")` is different from *defining* the function, and that calling the function is unindented, so is outside the function definition.

For reference, here is another example of a function:

```python
def add(x, y):  # the function add() takes in 2 inputs: x and y
  return x + y  # the function add() returns the value: x + y
  
a = add(2, 5)   # call the function add(), specifying x = 2 and y = 5
print(a)        # show that the variable 'a' now has a value of 7
```

First, notice that we now have 2 arguments (function inputs), `x` and `y`. Second, notice that instead of `print()`, we have `return x + y`. What this means is that the function `add(x, y)` will now return a value. Because the function returns a value, we can do variable assignment of a function, `a = add(2, 5)`, and `a` now has a value of `7`. To recap, `2` and `5` are _set_ into `x` and `y` respectively, and `x + y`, i.e. `7`, is returned and assigned to the variable `a`.

## 4 Data Structures: List, Tuple, Dictionary, Set

What if we want to manipulate many variables and values (think hundreds or thousands) at a time? Because we want to easily access and modify variables, we use data structures instead of manually defining and assigning thousands of individual variables.

### List

For example, take the example of having 3 variables:

```python
name1 = "Jet"
name2 = "Jack"
name3 = "Jatt"
```

Instead of 3 individual variable assignments, let's use a list:

```python
name_list = ["Jet", "Jack", "Jatt"]
```

A list of elements is encapsulated by square brackets `[]`, and elements delimited (separated) by `,` commas. To access elements of the list, we use _list indexing_:

```python
name_list = ["Jet", "Jack", "Jatt"]
print(name_list[0])  # prints "Jet"
print(name_list[1])  # prints "Jack"
print(name_list[2])  # prints "Jatt"
print(name_list[-1])  # prints the last element of the list, "Jatt"
```

List indexing refers to accessing the element of a list by its _index_. Notice that the first element is indexed by `0`. In most programming languages, the first index is `0` by design. While seeming cryptic, you just need to get used to it. Just remember that indices start from `0` because "off-by-1" indexing errors are common among beginner programmers. 

You can also do list _slicing_, which accesses subsequences of elements of a list:

```python
number_list = ['a', 'b', 'c', 'd', 'e', 'f']
print(number_list[0:3])  # prints ['a', 'b', 'c']
print(number_list[:3])   # prints ['a', 'b', 'c']
print(number_list[-3:])  # prints ['d', 'e', 'f']
```

List slicing works by having a *start* and *end* index, separated by a `:` colon, e.g. `number_list[0:3]` means _access elements starting from index 0 and ending at (up to but not including) index 3_. This means that although `'d'` is at index 3, it is not included. This is also by design, which is also a common source of beginner errors. Just get used to it.

If you want elements starting from the first index, you can drop the start index, as in `number_list[:3]`. Likewise, if you want the elements all the way to the last index, you can drop the end index, as in `number_list[-3:]`.

To modify the element of a list, you can just do assignment by index:

```python
number_list = ['a', 'b', 'c', 'd', 'e', 'f']
number_list[1] = 'B'  # set the element at index 1 as 'B'
number_list[4] = 'E'  # set the element at index 4 as 'E'
print(number_list)    # prints ['a', 'B', 'c', 'd', 'E', 'f']
```

### Tuple

Tuples are essentially the same as lists, with the exception that you cannot modify elements of a tuple after creating the tuple. That means that the following code will output an error:

```python
number_list = ('a', 'b', 'c')
number_list[0] = 'A'  # TypeError: 'tuple' object does not support item assignment
```

Notice that running the above code results in a _type_ error: `'tuple' object does not support item assignment`. You cannot perform element assignment for tuples. Tuples encapsulate elements with parenthesis `()`.

Why don't tuples support element assignment? Why would I use a tuple instead of a list which supports element assignment? The property being able to modify values after creation is _mutability_, or the ability to _mutate_. In some cases in programming, you might want to ensure that certain variables are not changed after creation, and you can enforce that by using tuples as the data structure instead of lists. For example, a data point that is created (e.g. a new customer) may need to be created and destroyed as-is (with time created, date created, etc), and no modification to attributes of the data point is allowed.

Tuple indexing works the same as list indexing.

### Dictionary

Dictionaries are a type of data structure that associates a 'key' with a 'value'. While we reference elements of a list and tuple by its index, we reference elements of a dictionary by its key. Dictionaries encapsulate elements with curly braces `{}`:

```python
animal_counts = {'chicken': 3, 'eggs': 4, 'ducks': 0}
```

Each element in the dictionary is a key-value pair, where the key-value are associated using a `:` colon (e.g. `'chicken': egg` is a key-value pair). Dictionaries do not have an index and to access elements, use the key:

```python
animal_counts = {'chicken': 3, 'eggs': 4, 'ducks': 0}
print(animal_counts['chicken'])
print(animal_counts['duck'])
```

Note that you can only reference the value by its key, and not reference the key by its value. Dictionaries are especially useful when you want to provide a semantics when referencing values (in contrast to referencing by a meaningless index).

### Set

A set is a collection of unique elements, encapsulating elements using curly braces `{}` (like a dictionary). However, unlike a dictionary, it does not have a key-value pair, only elements:

```python
number_set = {1, 2, 3, 4, 4, 4}
print(number_set)  # prints {1, 2, 3, 4}
```

Notice that even though we specify multiple instances of `4`, a set automatically keeps a collection of unique elements only. This is useful when we want to count the number of unique elements in a collection.

## Booleans and Conditionals (If-Elif-Else)

### Boolean

We have learnt previously that booleans are a data type that is either a `True` or `False` value. In the while-loop, we learnt that we can use *boolean conditions* (e.g. `i < 5`) as well. Here are the boolean conditions:

```python
i = 5
print(i == 5)  # prints True
print(i > 5)   # prints False
print(i >= 5)  # prints True
print(i < 5)   # prints False
print(i <= 5)  # prints True
print(i != 5)  # prints False
```

Note that you cannot do `=<` or `=>`, you have to do `>=` and `<=`. We can also combine boolean conditions to create more complex boolean expressions using `and` and `or`.

```python
i = 5
j = 7
print(i < 5 or j >= 7)   # prints True
print(i < 5 and j >= 7)  # prints False
```

### Conditionals (If-Elif-Else)

We can adapt the code to adapt to different situations by using `if`, `elif` and `else` (in that order). If a boolean expression is true, the code run will enter that condition.

```python
age = 21
if age >= 18:
  print("Can learn driving!")
else:
  print("Cannot learn driving.)
```

The above code will print `"Can learn driving!"` if the condition `age >= 18` is `True`, which it is. The `else` statement is a *catch-all* condition that will run if all the above `if`s and `elif`s are `False`. For example:

```python
age = 17
if age >= 21:
  print("Legal to smoke.")
elif age >= 18:
  print("Can learn driving!")
else:
  print("Go back to school!")
```

Notice that since `age >= 21` and `age >= 18` are both `False`, the default code body under `else` will be run. One common beginner mistake is confusing the use of `if` and `elif`, especially in the middle. Consider these 2 code snippets:

```python
age = 23
if age >= 21:
  print("Legal to smoke.")
if age >= 18:  # we use 'if'
  print("Can learn driving!")
else:
  print("Go back to school!")
```

```python
age = 23
if age >= 21:
  print("Legal to smoke.")
elif age >= 18:  # we use 'elif'
  print("Can learn driving!")
else:
  print("Go back to school!")
```

Notice that for the first code snippet uses `if` and the second uses `elif` in the middle condition. The first code snippet will print both `"Legal to smoke."` and `"Can learn driving!"` while the second snippet will only print `"Legal to smoke."`. This is because the code body of `elif` will only be run if the above `if` conditions are `False`.


## For-loops and While-loops

If we want to run the same (or similar) code multiple times, one elegant way of doing so, rather than copy-pasting the same line of code over and over again, is by using loops.

### For-loop

The for-loop is an easy way to specify doing something $N$ number of times:

```python
for i in range(5):  # repeats the code body 5 times
  print("Hello!")   # prints "Hello!" 5 times
```

There are 3 things to note when creating a for-loop:

1. The syntax starts with `for i in range(5)`, where `i` can be any variable name you define (you can try changing it to `n` or `count`), and `5` is the number of times you want to loop.
2. The for-loop ends with a `:` colon on the right.
3. The code body of what you want to loop must be indented with tabs.

The variable `i` is a counter variable. In `for i in range(5)`, it takes the values `[0, 1, 2, 3, 4]` respectively for each loop. This is because `range(5)` is an iterable (something that is able to be iterated):

```python
print(list(range(5)))  # prints [0, 1, 2, 3, 4]
```

The `range()` function can also take in multiple inputs. The following 3 for-loops **do the same thing**.

```python
for i in range(5):  # End at (but not including) 5
  print(i)
  
for i in range(0, 5):  # Start at 0, end at (but not including) 5
  print(i)
  
for i in range(0, 5, 1):  # Start at 0, end at (but not including) 5, incrementing i by 1 every loop
  print(i)
```

Therefore, if you want `i` to iterate *backwards* from `7` to `3`, you can do this:

```python
for i in range(7, 2, -1):  # Start at 7, end at (but not including) 2, incrementing i by -1 every loop
  print(i)
```

### While-loop

A while-loop is another (more flexible) way to specify loops. This is how a standard while-loop looks like:

```python
i = 0
while i < 5:  # Enter the loop's code body as long as i < 5 is true, and will exit when i increases to 5
  print(i)
  i += 1  # Increase i by 1
```

There are 3 things to note for while-loops:

1. Initialise a *counter* variable, e.g. `i`.
2. Specify a continue condition, e.g. `i < 5`, that specifies whether to enter the loop's code body.
3. Increment the counter variable, i.e. `i`, and ensure that the continue condition is eventually false, by which the loop will be exited.

A very common beginner mistake is an infinite-loop, where the *continue condition* is never false. This could be because you forget to increment the counter variable `i`. Regardless, when this happens, click the terminal and press `Ctrl` + `L`, which will interrupt the terminal and bring the code run to a stop.

A while loop is much more flexible because you can flexibly change the *continue condition* to be a more complex boolean expression.

### Conclusion

We have gone through the following concepts:

1. Print and Comments
2. Variables and Data Types
3. Operators
4. Functions
5. 4 Data Structures: List, Tuple, Dictionary, Set
6. Booleans and Conditionals
7. For-loops and While-loops

With this quickstart in programming, you are ready to Google search your way through programming tasks.