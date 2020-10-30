Kunyu An

Credit Ranking Model

Files Included
-----------------
logistic.py
logistic_sk.py
accepts.csv
reject.csv
woe.py
readme.txt

Project Description
-----------------
This program is to realize a credit ranking system with python.
This project contains two main programs with many components.
To run this program, follow the command below.


logistic.py
-----------------
This program can realize a credit ranking system with logistic regression method
using "stats.model" package.
Before running the program, please change line 10 to you own path(for windows users),
or you can just comment out line 10(for mac users).

to run: python3 logistic.py

credit_ranking.py
-----------------
This program can realize a credit ranking system with both logistic regresssion and 
decesion tree method, using the "sklearn" package.
Before running the program, please change line 22 to you own path(for windows users),
or you can just comment out line 22(for mac users).

--Logistic Regression--
to run: python3 credit_ranking.py
        then follow the prompt by type "l" in the terminal

--Decision Tree--
to run: python3 credit_ranking.py
        then follow the prompt by type "d" in the terminal


NOTE
-----------------
logistic.py is a test version for this project. For full version, please find credit_ranking.py.

Bugs
-----------------
Haven't found yet.
