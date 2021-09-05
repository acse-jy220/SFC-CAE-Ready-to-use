import numpy as np
import vtk, vtktools
import sys

# python MSE_velocity.py file_original.vtu file_reconstructed.vtu output_vtu_name.vtu
 
# 1. this will print the MSE of the velocity from two vtu files (which have the SAME MESH)
# 2. it then creates a vtu with the velocity field from file_original.vtu, the velocity field from file_reconstructed.vtu, and the pointwise error in velocity 

# 1. the MSE  
file1 = sys.argv[1]
file2 = sys.argv[2]
output_vtu_name = sys.argv[3]

vtu1 = vtktools.vtu(file1)
vtu2 = vtktools.vtu(file2)

velocity1 = vtu1.GetField('Velocity')
velocity2 = vtu2.GetField('Velocity')

nNodes = velocity1.shape[0]
print('nNodes', nNodes)

# assumes 2D problem
u1 = velocity1[:,0] # u velocity of file 1
v1 = velocity1[:,1] # v velocity of file 1
u2 = velocity2[:,0] # u velocity of file 2
v2 = velocity2[:,1] # v velocity of file 2

# here's one way of defining the MSE of the velocity
mse_velocity = np.sum(np.square(u1-u2) + np.square(v1-v2)) / nNodes

print('mse error in velocity', mse_velocity)

# 2. the pointwise velocity

#function to get a clean vtu file
def get_clean_vtk_file(filename):
    "Removes fields and arrays from a vtk file, leaving the coordinates/connectivity information."
    vtu_data = vtktools.vtu(filename)
    clean_vtu = vtktools.vtu()
    clean_vtu.ugrid.DeepCopy(vtu_data.ugrid)
    fieldNames = clean_vtu.GetFieldNames()
# remove all fields and arrays from this vtu
    for field in fieldNames:
        clean_vtu.RemoveField(field)
        fieldNames = clean_vtu.GetFieldNames()
        vtkdata=clean_vtu.ugrid.GetCellData()
        arrayNames = [vtkdata.GetArrayName(i) for i in range(vtkdata.GetNumberOfArrays())]
    for array in arrayNames:
        vtkdata.RemoveArray(array)
    return clean_vtu

new_vtu = get_clean_vtk_file(file1) 
new_vtu.filename = output_vtu_name 
new_vtu.AddField('Original', velocity1)
new_vtu.AddField('Reconstructed', velocity2)
new_vtu.AddField('Pointwise Error', velocity2 - velocity1[:,:2]) # 
new_vtu.Write()

