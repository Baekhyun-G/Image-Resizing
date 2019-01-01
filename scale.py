import matplotlib.pyplot as plt 
import scipy.misc
import matplotlib.mlab as mlab
import numpy as np
from scipy import signal
import math
from PIL import Image
import random
import ntpath

image=input("Relative path of file: ")
face1 = np.array(Image.open(image).convert('L'))

wh=input("Scale height or width? (w or h) :")
try:
	if (wh=='h' or wh=='height'):
		face1=np.transpose(face1)
	m,n=face1.shape
except e:
	print("If you wish to scale the height enter 'h' or 'height' else enter 'w' or 'width'")
	sys.exit(1)
try:
	opt=input("upscale/downscale (u or d) : ")
except e:
	print("If you wish to downscale the image enter 'd' else enter 'u'")
	sys.exit(1)
if wh=="w":
	print("The width of the image is: "+str(n))
	scale=int(input("Final width of image: "))
	if opt=='u':
		while (scale<n or scale>=2*n):
			print("The final width of the image should be LARGER than the width of the image!!!")
			scale=int(input("Final width of image: "))
	else :
		while (scale>=n or scale<2):
			print("The final width of the image should be SMALLER than the width of the image!!!")
			scale=int(input("Final width of image: "))

else:
	print("The height of the image is: "+str(n))
	scale=int(input("Final height of image: "))
	if opt=='u':
		while (scale<n or scale>=2*n):
			print("The final height of the image should be LARGER than the height of the image!!!")
			scale=int(input("Final height of image: "))
	else :
		while (scale>=n or scale<2):
			print("The final height of the image should be SMALLER than the height of the image!!!")
			scale=int(input("Final height of image: "))

face1=face1.flatten()
face1=np.array([int(i) for i in face1])
face1=face1.reshape(m,n)
face=face1.tolist()
mn=n
kx=np.array([[1,0,-1]])
ky=np.array([[1],[0],[-1]])
gx=np.absolute(signal.convolve2d(face,kx,boundary='symm',mode='same'))
gy=np.absolute(signal.convolve2d(face,ky,boundary='symm',mode='same'))
g=(gx+gy)
sumc=np.argsort(np.sum(g, axis=0))[:30]
face2=np.delete(face1,sumc,1)

if opt=='d':
	iter1=n-scale
	for intl in range(iter1):
		gx=np.absolute(signal.convolve2d(face,kx,boundary='symm',mode='same'))
		gy=np.absolute(signal.convolve2d(face,ky,boundary='symm',mode='same'))
		g=(gx+gy)

		dp=np.zeros((m,n))

		dp[0]=g[0]
		def fun(i,j,dp):
			if j==0:
				return min(dp[i-1][j],dp[i-1][j+1])+g[i][j]
			elif j==n-1:
				return min(dp[i-1][j],dp[i-1][j-1])+g[i][j]
			else:
				return min(dp[i-1][j],dp[i-1][j+1],dp[i-1][j-1])+g[i][j]
		for i in range(m):
			if i==0:
				continue
			else :
				dp[i]=np.array([fun(i,j,dp) for j in range(n)])
		ind=np.zeros(m,int)
		for i in range(m):
			if i==0:
				ind[m-1]=list(dp[m-1]).index(min(dp[m-1]))
			else :
				if ind[m-i]==0:
					if g[m-i-1][0]<g[m-i-1][1]:
						ind[m-i-1]=0
					else:
						ind[m-i-1]=1
				elif ind[m-i]==n-1:
					if g[m-i-1][n-1]<g[m-i-1][n-2]:
						ind[m-i-1]=n-1
					else:
						ind[m-i-1]=n-2
				else:
					if g[m-i-1][ind[m-i]]<min(g[m-i-1][ind[m-i]+1],g[m-i-1][ind[m-i]-1]):
						ind[m-i-1]=ind[m-i]
					elif g[m-i-1][ind[m-i]-1]<min(g[m-i-1][ind[m-i]+1],g[m-i-1][ind[m-i]]):
						ind[m-i-1]=ind[m-i]-1
					elif g[m-i-1][ind[m-i]+1]<min(g[m-i-1][ind[m-i]],g[m-i-1][ind[m-i]-1]):
						ind[m-i-1]=ind[m-i]+1
		for i in range(m):
			del face[i][ind[i]]
		n=n-1
	if (wh=='h' or wh=='height'):
		face=np.transpose(face)
	result = Image.fromarray(np.array(face).astype(np.uint8))
	result.save("scaled_"+ntpath.basename(image))
	# fig = plt.figure(frameon=False)
	# fig.set_size_inches(m,n)
	# ax = plt.Axes(fig, [0., 0., 1., 1.])
	# ax.set_axis_off()
	# fig.add_axes(ax)
	# ax.imshow(face, cmap='gray')
	

	# fig.savefig("scaled_"+ntpath.basename(image),bbox_inches='tight')

if opt=='u':
	gx=np.absolute(signal.convolve2d(face,kx,boundary='symm',mode='same'))
	gy=np.absolute(signal.convolve2d(face,ky,boundary='symm',mode='same'))
	g=np.sqrt((np.square(gx)+np.square(gy)))

	dp=np.zeros((m,n))

	dp[0]=g[0]
	def fun(i,j,dp=dp):
		if j==0:
			return min(dp[i-1][j],dp[i-1][j+1])+g[i][j]
		elif j==n-1:
			return min(dp[i-1][j],dp[i-1][j-1])+g[i][j]
		else:
			return min(dp[i-1][j],dp[i-1][j+1],dp[i-1][j-1])+g[i][j]
	for i in range(m):
		if i==0:
			continue
		else :
			dp[i]=np.array([fun(i,j) for j in range(n)])
	iter1=scale-n
	ind=np.zeros((m,iter1),int)
	# print(dp[m-1])
	def randi():
		a=random.uniform(0,1)
		if a<0.5:
			return 0
		else :
			return 1
	def randiv():
		a=random.uniform(0,1)
		if a<0.33:
			return 0
		elif (a>=0.33 and a<0.67) :
			return 1
		else:
			return 2

	sumc=np.argsort(dp[m-1])[:iter1]
	for k in range(iter1):
		for i in range(m):
			if i==0:
				ind[m-1][k]=sumc[k]
			else:
				if int(ind[m-i][k])==0:
					
					if dp[m-i-1][0]<dp[m-i-1][1]:
						ind[m-i-1][k]=0
					elif dp[m-i-1][0]>dp[m-i-1][1]:
						ind[m-i-1][k]=1
					else :
						ab=randi()
						# print(ab)
						if ab==1:
							ind[m-i-1][k]=0
						else :
							ind[m-i-1][k]=1


				elif int(ind[m-i][k])==(n-1):
					if dp[m-i-1][n-1]<dp[m-i-1][n-2]:
						ind[m-i-1][k]=n-1
					elif dp[m-i-1][n-1]>dp[m-i-1][n-2]:
						ind[m-i-1][k]=n-2
					else :
						ab=randi()
						# print(ab)
						if ab==1:
							ind[m-i-1][k]=n-1
						else :
							ind[m-i-1][k]=n-2
				else:
					p=dp[m-i-1][ind[m-i][k]]
					q=dp[m-i-1][ind[m-i][k]+1]
					r=dp[m-i-1][ind[m-i][k]-1]
					if p<min(q,r):
						ind[m-i-1][k]=ind[m-i][k]
					elif r<min(q,p):
						ind[m-i-1][k]=ind[m-i][k]-1
					elif q<min(p,r):
						ind[m-i-1][k]=ind[m-i][k]+1
					elif p==q:
						ab=randi()
						if ab==1:
							ind[m-i-1][k]=ind[m-i][k]
						else :
							ind[m-i-1][k]=ind[m-i][k]+1
					elif r==q:
						ab=randi()
						if ab==1:
							ind[m-i-1][k]=ind[m-i][k]-1
						else :
							ind[m-i-1][k]=ind[m-i][k]+1
					elif p==r:

						ab=randi()
						if ab==1:
							ind[m-i-1][k]=ind[m-i][k]
						else :
							ind[m-i-1][k]=ind[m-i][k]-1
					else :
						ab=randiv()
						if ab==1:
							ind[m-i-1][k]=ind[m-i][k]
						elif ab==0 :
							ind[m-i-1][k]=ind[m-i][k]-1
						else :
							ind[m-i-1][k]=ind[m-i][k]+1





	# print(ind)
	for i in range(m):
		for kl in sorted(ind[i],reverse=True):


			face[i].insert(kl,face[i][kl])


	# face=face[:,:(int(n/2))]
	# plt.figure(figsize=[5,5])


	# plt.subplot(221)

	if (wh=='h' or wh=='height'):
		face=np.transpose(face)
	result = Image.fromarray(np.array(face).astype(np.uint8))
	result.save("scaled_"+ntpath.basename(image))
	# plt.subplot(222)

	# plt.imshow(face,cmap='gray')
	# # plt.subplot(224)

	# # plt.imshow(face2,cmap='gray')
	# plt.show()