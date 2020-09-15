from sklearn.metrics import accuracy_score

y_true = [[  1,   1,	1,	1,	1], 
            [1,   1,	1,	1,	1]
         ]
y_pred = [[1, 0,	1,	0,	1], 
          [1, 0,	1,	0,	0]]

print(accuracy_score(y_true, y_pred, False))