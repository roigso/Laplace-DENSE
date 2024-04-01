"""
This module contains the mesh class implemented for tetrahedra.
"""
import numpy as np
import meshio


class Mesh:
    """
    Class that contains the mesh.

    Args:    
        filename (str): the path and filename of the file with the mesh. The file format must be compatible with meshio.
    
    Attributes:
        verts (array): a numpy array that contains all the nodes of the mesh.
                       verts[i,j], where i is the node index and j=[0,1,2] is the coordinate (x,y,z).
        connectivity (array): a numpy array that contains all the connectivity of the tetrahedra of the mesh.
                              connectivity[i,j], where i is the tetrahedron index and j=[0,1,2,3] is node index.                
    """


    def __init__(self, filename  = None, verts = None, connectivity = None):
        if filename is not None:
            verts, connectivity = self.load_mesh(filename)
        self.data         = None # because there is no mesh with data 
        self.verts        = np.array(verts)
        self.connectivity = np.array(connectivity)        



    def load_mesh(self, filename):
        """
        This function reads a mesh file using meshio.
        Args:
            filename (str): the path and filename of the mesh file. The mesh must have only one type of element (tetrahedra).
        """
        self.data    = meshio.read(filename)
        verts        = self.data.points # verts[i,j], where i is the node index and j=[0,1,2] is the coordinate (x,y,z)
        connectivity = list(self.data.cells_dict.values())[0] # it assumes there is one type of element (tetrahedra) 
                                                              # TODO: check this before extracting the values

        return verts, connectivity



    def writeVTU(self, filename, verts = None, connectivity = None, scalars = {}, vectors = {}):
        
        if verts is None:
            verts = self.verts if self.data is None else self.data.points 
                
        connectivity = self.connectivity if connectivity is None else connectivity 

        # currently only works with tetrahedra
        # scalars and vectors goes to point_data: scalars is an array with the nodal values, vectors is a n_nodes x dim (3 in 3D) array 
        mesh_write = meshio.Mesh(points = verts, cells = [("tetra",connectivity)], point_data = scalars | vectors)
                    # also meshio.Mesh() has the options to save cell_data and field_data 
        meshio.write(filename, mesh_write)



    def Bmatrix(self, element):
        nodeCoords = self.verts[self.connectivity[element]]

        # NOTE: The definition of the parent tetrahedron by Hughes - "The Finite Element Method" (p. 170) is different from the 
        # local order given by VTK (http://victorsndvg.github.io/FEconv/formats/vtk.xhtml) 
        # Here we follow the VTK convention
        x1 = nodeCoords[0][0]; x2 = nodeCoords[1][0]; x3 = nodeCoords[2][0]; x4 = nodeCoords[3][0]
        y1 = nodeCoords[0][1]; y2 = nodeCoords[1][1]; y3 = nodeCoords[2][1]; y4 = nodeCoords[3][1]
        z1 = nodeCoords[0][2]; z2 = nodeCoords[1][2]; z3 = nodeCoords[2][2]; z4 = nodeCoords[3][2]

        x14 = x1 - x4; x34 = x3 - x4; x24 = x2 - x4
        y14 = y1 - y4; y34 = y3 - y4; y24 = y2 - y4
        z14 = z1 - z4; z34 = z3 - z4; z24 = z2 - z4

        detJ = x14 * (y34 * z24 - z34 * y24) - y14 * (x34 * z24 - z34 * x24) + z14 * (x34 * y24 - y34 * x24)

        Jinv_11 =        y34 * z24 - y24 * z34 ; Jinv_12 = -1. * (x34 * z24 - x24 * z34); Jinv_13 =        x34 * y24 - x24 * y34
        Jinv_21 = -1. * (y14 * z24 - y24 * z14); Jinv_22 =        x14 * z24 - x24 * z14 ; Jinv_23 = -1. * (x14 * y24 - x24 * y14)
        Jinv_31 =        y14 * z34 - y34 * z14 ; Jinv_32 = -1. * (x14 * z34 - x34 * z14); Jinv_33 =        x14 * y34 - x34 * y14

        B_def = np.array([[1., 0., 0., -1.],
                          [0., 0., 1., -1.],
                          [0., 1., 0., -1.]])
        
        Jinv = np.array([[Jinv_11, Jinv_12, Jinv_13],
                         [Jinv_21, Jinv_22, Jinv_23],
                         [Jinv_31, Jinv_32, Jinv_33]])
        
        B = np.dot(Jinv.T,B_def)

        return B, detJ



    def gradient(self, element, u):
        # in linear finite elements, the gradient is constant inside the element
        B, detJ = self.Bmatrix(element)
        grad    = 1. / detJ * np.dot(B, u)
        return grad # 3 x 1
        # TODO: check the gradient



    def StiffnessMatrix(self,B,J):
        # stiffness matrix for tetrahedra
        return np.dot(B.T, B) / (6. * J)
    


    def MassMatrix(self,J):
        # mass matrix for tetrahedra
        return np.array([[1. / 60., 1. / 120., 1. / 120., 1. / 120.],
                        [1. / 120., 1. / 60. , 1. / 120., 1. / 120.],
                        [1. / 120., 1. / 120., 1. / 60. , 1. / 120.],
                        [1. / 120., 1. / 120., 1. / 120., 1. / 60. ]])*J



    def computeLaplace(self, nodes, nodeVals, filename = None):
        if filename is None:
            return self.computePoisson(nodes, nodeVals, f_dom = 0., filename = None)
        else:
            return self.computePoisson(nodes, nodeVals, f_dom = 0., filename = filename)



    def computePoisson(self, nodes, nodeVals, f_dom, filename = None):
        """
        This function solves the Poisson problem.
        Args:
            nodes (array): nodes with Dirichlet boundary conditions.
            nodeVals (array): values for the nodes with Dirichlet boundary conditions.
            f_dom: represents the given function in the domain, it can be a Python function depending on the coordinates, or a float.
                   if float:    f_dom = float
                   if function: f_dom = f(coords), where coords = [x,y,z] are the coordinates.
        """

        nNodes = self.verts.shape[0]
        nElem = self.connectivity.shape[0]
        
        K = np.zeros((nNodes,nNodes))
        M = np.zeros((nNodes,nNodes))
        
        f_dom_nodes = np.zeros((nNodes,1)) # a vector with f_dom() evaluated in the nodes
        
        activeNodes = list(range(nNodes))
        for known in nodes:
            activeNodes.remove(known)
            
        jActive, iActive = np.meshgrid(activeNodes, activeNodes)
        jKnown, iKnown   = np.meshgrid(nodes, activeNodes)
        
        Js = np.zeros((nElem,1))
        
        for k,tri in enumerate(self.connectivity):
            j, i = np.meshgrid(tri,tri)
            B, J = self.Bmatrix(k)
            Js[k] = J # not used

            k = self.StiffnessMatrix(B,J)
            K[i, j] += k

            m        = self.MassMatrix(J)
            M[i, j] += m
        
        for node_ind, node_coord in enumerate(self.verts):
            f_dom_nodes[node_ind] = f_dom(node_coord) if callable(f_dom) else f_dom

        F = np.dot(M, f_dom_nodes)
        
        T = np.linalg.solve(K[iActive, jActive],F[activeNodes,0]-np.dot(K[iKnown, jKnown],nodeVals))
        
        Tglobal = np.zeros(nNodes)
        
        Tglobal[activeNodes] = T
        Tglobal[nodes] = nodeVals

        if filename is not None:
            self.writeVTU(filename, scalars = {"Tglobal": Tglobal})
            # self.writeVTU(filename, self.verts, self.connectivity, scalars=Tglobal, None)
            
        return Tglobal
    


    def computeLaplacian(self):
        nNodes = self.verts.shape[0]
        K = np.zeros((nNodes,nNodes), dtype = "float16")
        M = np.zeros((nNodes,nNodes), dtype = "float16")
        
        for k,tri in enumerate(self.connectivity): # k: element, tri: nodes of the element
            j, i     = np.meshgrid(tri,tri)
            B, detJ  = self.Bmatrix(k)

            k        = self.StiffnessMatrix(B, detJ)
            K[i,j] += k

            m        = self.MassMatrix(detJ)
            M[i,j]  += m
           
        return K, M
        