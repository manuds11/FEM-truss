import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
    Terminology: Abreviations and Acronyms


    Material Properties:
        (E = Young Modulus, A = Cross section Area)
    



    # ______ INPUT VARIABLES _________
    
    
    
    # ______ PROBLEM VARIABLES _______

    
"""

# **************** Classes *****************

class Model():
    def __init__(self, Model_nodes ,Model_elements):
        self.nodes = Model_nodes
        self.elements = Model_elements
        self.node_DOF = len(Model_nodes[0].GetNodeCoords())
        self.K_global = np.zeros((np.shape(self.nodes)[0] * self.node_DOF, np.shape(self.nodes)[0] * self.node_DOF))
        # self.K_reduced = self.K_global
        self.restrictions = []
        self.AppliedF_full_asColumns = []
        self.K_reduced = None
        
        self.fig_pre_calc = plt.figure(1)
        self.fig_post_calc = plt.figure(2)
    
    def GetModelNodes(self):
        return self.nodes
    def GetModelElements(self):
        return self.elements
    
    def GetNodes_IndexAndCoords_asDict(self):
        Nodes_IndexAndCoords = []
        for node in self.GetModelNodes():
            IndexAndCoords = {
                'Id': node.GetNodeId(), 
                '(X, Y)': node.GetNodeCoords(),
                }
            Nodes_IndexAndCoords.append(IndexAndCoords)
        return Nodes_IndexAndCoords
    
    def GetElem_IndexAndNodes_AsDict(self):
        elem_IndexAndNodes = []
        for elem in self.GetModelElements():
            elem_nodes_index = elem.GetElem_NodesIndex()
            IndexAndNodes = {
                'Id': elem.GetElemIndex(),
                'Nodes Index': elem.GetElem_NodesIndex(),
                }
            elem_IndexAndNodes.append(IndexAndNodes)
        
        return elem_IndexAndNodes

    def Assemble_K_global(self, node_DOF = 2):        
        for element in self.elements:
            node1, node2 = element.GetElem_NodesIndex()
            K_elem = element.Get_K_elem()
            for i in range(np.shape(K_elem)[0]):
                for j in range(np.shape(K_elem)[0]):
                    if ((i + 1) <= node_DOF and (j + 1) <= node_DOF):
                        pivot = (node1 - 1) * node_DOF
                        self.K_global[pivot + i][pivot + j] += K_elem[i][j]
                    elif ((i + 1) > node_DOF and (j + 1) > node_DOF):
                        pivot = (node2 - 1) * node_DOF
                        self.K_global[pivot + i - 2][pivot + j - 2] += K_elem[i][j]
                    elif ((i + 1) <= node_DOF and (j + 1) > node_DOF):
                        pivot_i = (node1 - 1) * node_DOF
                        pivot_j = (node2 - 1) * node_DOF
                        self.K_global[pivot_i + i][pivot_j + j - 2] += K_elem[i][j]
                    elif ((i + 1) > node_DOF and (j + 1) <= node_DOF):
                        pivot_i = (node2 - 1) * node_DOF
                        pivot_j = (node1 - 1 ) * node_DOF
                        self.K_global[pivot_i + i - 2][pivot_j + j] += K_elem[i][j]
                        
        return self.K_global
 
    def GetRestrictions_asMatrix(self):
        restricts_asMatrix = []
        for Node in self.nodes:
            restricts_asMatrix.append(Node.GetNodeRestrictions())
        return restricts_asMatrix

    def ReshapeNodeRestrictions_toColumns(self):
        Restricts_asColumn = []
        for Node in self.nodes:
            Node_restrictions = Node.GetNodeRestrictions()
            Restricts_asColumn.append(Node_restrictions[0])
            Restricts_asColumn.append(Node_restrictions[1])
        return Restricts_asColumn
    
    def GetRestrictions_asColumn(self):        
        self.restrictions = self.ReshapeNodeRestrictions_toColumns()
        return self.restrictions
  
    def Compute_K_reduced(self):
        Lines_to_erase = []
        self.K_reduced = self.K_global
        for i in range(len(self.restrictions)):
            if self.restrictions[i]:
                Lines_to_erase.append(i)
        for i in reversed(Lines_to_erase):  # Erasing the ith+1 row and column.
            self.K_reduced = np.delete(self.K_reduced, i, 0); self.K_reduced = np.delete(self.K_reduced, i, 1) 
        return self.K_reduced
    
    def GetAppliedForces_asMatrix(self):
        AppliedF_asMatrix = []
        for Node in self.nodes:
            AppliedF_asMatrix.append(Node.GetNodeAppliedForces())
        return AppliedF_asMatrix
    
    def ReshapeAppliedForces_asColumns(self):
        AppliedF_asColumns = []
        for Node in self.nodes:
            Node_Applied_Forces = Node.GetNodeAppliedForces()
            AppliedF_asColumns.append(Node_Applied_Forces[0])
            AppliedF_asColumns.append(Node_Applied_Forces[1])
        return AppliedF_asColumns
    
    def GetAppliedForces_asColumns(self):
        self.AppliedF_full_asColumns = self.ReshapeAppliedForces_asColumns()
        return self.AppliedF_full_asColumns
    
    
    def ComputeF_reduced(self):
        self.F_reduced= self.GetAppliedForces_asColumns().copy()
        for i in range( len(self.restrictions) - 1, -1, -1 ):
            if self.restrictions[i]: self.F_reduced.pop(i)        
        return self.F_reduced
    
    def ComputeDisplacements(self):
        self.disp_reduced = np.linalg.inv(self.K_reduced) @ np.array(self.F_reduced)
        return self.disp_reduced
    def ConstructDisp_full(self):
        self.disp_full = []; j = 0
        for i in range(len(self.restrictions)):
            if self.restrictions[i]:   self.disp_full.append(0.0)
            else: self.disp_full.append(self.disp_reduced[j]); j += 1;      
        return np.array(self.disp_full) 
    
    def SetNodesDisp(self):
        node_DOF = self.node_DOF; i = 0
        for node in self.nodes:
            node.SetNodeDisp( (self.disp_full[i*node_DOF], self.disp_full[i*node_DOF+1]) )
            i += 1
    
   
    def ComputeStrains(self):
        self.Elem_strains = [];
        for elem in self.elements:
            self.Elem_strains.append(elem.CalculateElemStrain())
        return self.Elem_strains
    def ComputeStresses(self):
        self.Elem_stresses = []
        for elem in self.elements:
            self.Elem_stresses.append(elem.CalculateElemStress())
        return self.Elem_stresses
    
    
    # Display methods
    
    def PlotNodes(self):
        plt.figure(1)
        Nodes_IndexAndCoords = self.GetNodes_IndexAndCoords_asDict()
        
        for node in Nodes_IndexAndCoords:
            plt.plot(node["(X, Y)"][0], node["(X, Y)"][1], marker = "o", markersize = 10, markeredgecolor = "black", markerfacecolor = "yellow")
        
    def PlotElements(self):
        plt.figure(1)
        Nodes_IndexAndCoords = self.GetNodes_IndexAndCoords_asDict()
        Elem_IndexAndNodes = self.GetElem_IndexAndNodes_AsDict()
        for elem in Elem_IndexAndNodes:
            elem_node1_id, elem_node2_id = elem["Nodes Index"]
            Node_1_found = [it for it in Nodes_IndexAndCoords if it["Id"] == elem_node1_id]
            Node_2_found = [it for it in Nodes_IndexAndCoords if it["Id"] == elem_node2_id]
            A = np.array([Node_1_found[0]["(X, Y)"], Node_2_found[0]["(X, Y)"]])
            plt.plot(A[:, 0], A[:, 1], color = 'black'  )

    def PlotRestrictions(self):
        plt.figure(1)
        for node in self.GetModelNodes():
            if node.GetNodeRestrictions()[0]:
                plt.plot(node.GetNodeCoords()[0], node.GetNodeCoords()[1], 
                         marker = "^", markersize = 15, markeredgecolor = "black", markerfacecolor = "red", )
            if node.GetNodeRestrictions()[1]:
                plt.plot(node.GetNodeCoords()[0], node.GetNodeCoords()[1], 
                         marker = ">", markersize = 15, markeredgecolor = "black", markerfacecolor = "red", )
            
    def PlotForces(self):
        plt.figure(1)
        for node in self.GetModelNodes():
            F_vec = node.GetNodeAppliedForces()
            if F_vec != (0.0, 0.0):
                plt.quiver(node.GetNodeCoords()[0], node.GetNodeCoords()[1], F_vec[0], F_vec[1], color = 'blue', )

    def PlotStructureDisp_Exagerated(self, scale = 1000):
        plt.figure(2)
        Model_nodes = self.GetModelNodes()
        for elem in self.GetModelElements():
            node1_id, node2_id = elem.GetElem_NodesIndex()
            node1_X_exagerated, node1_Y_exagerated = Model_nodes[node1_id - 1].GetNodeFinalPositionExagerated(scale)
            node2_X_exagerated, node2_Y_exagerated = Model_nodes[node2_id - 1].GetNodeFinalPositionExagerated(scale)
            plt.plot([node1_X_exagerated, node2_X_exagerated], [node1_Y_exagerated, node2_Y_exagerated], color = 'black')



class Node():
    def __init__(self, global_id, coords, DOFs_restricted, F_applied):
        self.global_id = global_id
        self.coords = coords  # Tuple with the X, Y coords of the node in meters
        self.node_numberOfDOF = len(coords)
        self.DOFs_restricted = DOFs_restricted # If True, the DOF is restricted
        self.F_applied = F_applied # Force is to be specified in Newtons
        # Checks whether an external force is tried to be applied in a restricted DOF
        self.Disp = None
        j = 0;
        for restrict in self.DOFs_restricted:
            if restrict and self.F_applied[j] != 0:   
                raise Exception(""""An external force cannot be applied to a node restricted DOF.
                                \n\nYour Input:
                                    \nNode: {}
                                    \nRestrictions: {}
                                    \nExternal Forces: {}""".format(self.global_id, self.DOFs_restricted, self.F_applied))
            j += 1
                
    def GetNodeCoords(self):
        return self.coords    
    def getNodeNumberOfDOF(self):
        return self.node_numberOfDOF
    def GetNodeId(self):
        return self.global_id    
    def GetNodeRestrictions(self):
        return self.DOFs_restricted
    def GetNodeAppliedForces(self):
        return self.F_applied
    def SetNodeDisp(self, Disp):  # Disp is a tuple with (u, v) dispplacements
        self.Disp = Disp
    def GetNodeDisp(self):
        return self.Disp
    def GetNodeFinalPositionExagerated(self, scale):
        return (
            self.GetNodeCoords()[0] + self.GetNodeDisp()[0]*scale,
            self.GetNodeCoords()[1] + self.GetNodeDisp()[1]*scale
                )

    

class Truss_elem():
    # A truss element comprises 2 nodes, and one local DOF.
    
    
    def __init__(self, elem_id, node1, node2, E, A):
        self.elem_id = elem_id
        self.global_nodes_index = node1.GetNodeId(), node2.GetNodeId() #Tuple that indicates the global nodes index associated to the two local nodes that comprise the element         
        self.node1, self.node2 = node1, node2
        self.A = A     # in m^2
        self.E = E     # in N/m^2 (Pascals)
        node1_x, node1_y = node1.GetNodeCoords();
        self.node1_coords = node1_x, node1_y
        node2_x, node2_y = node2.GetNodeCoords();
        self.node2_coords = node2_x, node2_y
        # Calculated parameters
        self.L = np.sqrt( (node2_x - node1_x)**2 + (node2_y - node1_y)**2 ) 
        self.l_ij = (node2_x - node1_x) / self.L   # l_ij = cos(x, X)
        self.m_ij = (node2_y - node1_y) / self.L   # m_ij = cos(x, Y)
         
        self.k_elem = (self.A * self.E / self.L) * np.array([[1, -1],[-1, 1]])
         # Transformation matrix
        self.T = np.array([[self.l_ij, self.m_ij, 0, 0], [0, 0, self.l_ij, self.m_ij]])
        self.B = np.array([-1 / self.L, 1 / self.L]) # Matrix to compute element displacements
        self.K_elem = np.transpose(self.T) @ self.k_elem @ self.T
         
    def GetElemData(self):
        elem_data = {
            'Element Index': self.elem_id,
            'Global Node Index (node1 - node2)' : self.global_nodes_index,
            'Global Node Coords (x1, y1 - x2, y2) m': (self.node1_coords, self.node2_coords ),
            '(l_ij, m_ij ) = (cos(x, X), cos(x, Y))': (self.l_ij, self.m_ij), 
            'A (m^2)': self.A,
            'E (Pa)': self.E,
            'L (m)': self.L,
            'k_local (N*m^-2)': self.k_elem
        }
        elem_DataFrame = pd.DataFrame(elem_data)
        print(elem_DataFrame)
        return elem_data
    def GetElemIndex(self):
        return self.elem_id
    
    def GetElem_NodesIndex(self):
        return self.global_nodes_index
    
    
    def Get_K_elem(self):
        return self.K_elem
    def GetElemNodeDisps(self):
        d1_x, d1_y = self.node1.GetNodeDisp()
        d2_x, d2_y = self.node2.GetNodeDisp()
        return np.array([d1_x, d1_y, d2_x, d2_y])
    
    def CalculateElemStrain(self):
        Elem_disp = self.GetElemNodeDisps()
        self.Elem_strain = self.B  @ self.T @ Elem_disp
        
        return self.Elem_strain
    def CalculateElemStress(self):
        return self.E * self.Elem_strain


# ************* MODEL COMPUTATIONS *************


def PerformComputations(model):
    K_global = model.Assemble_K_global()
    Model_restrictions = model.GetRestrictions_asColumn()
    K_reduced = model.Compute_K_reduced()

        # Obtaining and reducing the vector of Applied Forces
    Model_Applied_Forces = model.GetAppliedForces_asColumns()
    Model_F_reduced = model.ComputeF_reduced()

        #Calculating the displacements of the Unrestricted DOFs
    Disp_reduced = model.ComputeDisplacements()
    Model_Disp_Full = model.ConstructDisp_full()
    model.SetNodesDisp()

        #Model Strains
    Model_strains = model.ComputeStrains()
    Model_stresses = model.ComputeStresses()

    return (K_global, Model_restrictions, K_reduced, 
            Model_Applied_Forces, Model_F_reduced, 
            Disp_reduced, Model_Disp_Full,
            Model_strains, Model_stresses)






         
        
        


