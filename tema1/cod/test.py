import cv2 as cv

A = [2, 3, 1, 1, 4]

dp1 = [0] * len(A)
dp2 = [False] * len(A)
dp2[len(A) - 1] = True
for i in reversed(range(0, len(A) - 2)):
    minim = float('inf')
    for j in range(i + 1, min(len(A), A[i] + i + 1)):
        if dp2[j] is True:
            dp2[i] = True
            minim = min(minim, dp1[j] + 1)
    dp1[i] = minim

if dp2[0] is False:
    print(-1)
print (dp1[0])