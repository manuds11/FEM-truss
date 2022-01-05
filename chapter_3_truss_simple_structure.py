import numpy as np
import pandas as pd


"""
    Terminology: Abreviations and Acronyms


    Material Properties:
        (E = Young Modulus, A = Cross section Area)
    



    # ______ INPUT VARIABLES _________
    
    
    
    # ______ PROBLEM VARIABLES _______

    
    """

# **************** Classes *****************

class Node():
    def __init__(self, global_id, coords, DOFs_restricted, F_applied):
        self.global_id = global_id
        self.coords = coords  # Tuple with the X, Y coords of the node in meters
        self.DOFs_restricted = DOFs_restricted # If True, the DOF is restricted
        self.F_applied = F_applied # Force is to be specified in Newtons
        # Checks wether an external force is tried to be applied in a restricted DOF
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


# ************** Model Functions ***************

def Assemble_K_global(Model_elements, Number_of_nodes, node_DOF = 2):
    K_global = np.zeros((Number_of_nodes*node_DOF, Number_of_nodes*node_DOF))
    for element in Model_elements:
        node1, node2 = element.GetElem_NodesIndex()
        K_elem = element.Get_K_elem()
        for i in range(np.shape(K_elem)[0]):
            for j in range(np.shape(K_elem)[0]):
                if ((i + 1) <= node_DOF and (j + 1) <= node_DOF):
                    pivot = (node1 - 1) * node_DOF
                    K_global[pivot + i][pivot + j] += K_elem[i][j]
                elif ((i + 1) > node_DOF and (j + 1) > node_DOF):
                    pivot = (node2 - 1) * node_DOF
                    K_global[pivot + i - 2][pivot + j - 2] += K_elem[i][j]
                elif ((i + 1) <= node_DOF and (j + 1) > node_DOF):
                    pivot_i = (node1 - 1) * node_DOF
                    pivot_j = (node2 - 1) * node_DOF
                    K_global[pivot_i + i][pivot_j + j - 2] += K_elem[i][j]
                elif ((i + 1) > node_DOF and (j + 1) <= node_DOF):
                    pivot_i = (node2 - 1) * node_DOF
                    pivot_j = (node1 - 1 ) * node_DOF
                    K_global[pivot_i + i - 2][pivot_j + j] += K_elem[i][j]
                    
    return K_global

def GetModelRestrictions(Model_nodes):

    Model_restrictions = []
    for Node in Model_nodes:
        Node_restrictions = Node.GetNodeRestrictions()
        Model_restrictions.append(Node_restrictions[0])
        Model_restrictions.append(Node_restrictions[1])
    
    return Model_restrictions



def Calculate_K_reduced(K_global, Model_restrictions):
    Lines_to_erase = []
    K_reduced = K_global
    for i in range(len(Model_restrictions)):
        if Model_restrictions[i]:
            Lines_to_erase.append(i)
    for i in reversed(Lines_to_erase):  # Erasing the ith+1 row and column.
        K_reduced = np.delete(K_reduced, i, 0); K_reduced = np.delete(K_reduced, i, 1)
    
    return K_reduced

def GetModelAppliedForces(Model_nodes):

    Model_Applied_Forces = []
    for Node in Model_nodes:
        Node_Applied_Forces = Node.GetNodeAppliedForces()
        Model_Applied_Forces.append(Node_Applied_Forces[0])
        Model_Applied_Forces.append(Node_Applied_Forces[1])
    
    return Model_Applied_Forces


def Calculate_F_reduced(Model_restrictions, Model_Applied_Forces):
    for i in range( len(Model_restrictions) - 1, -1, -1 ):
        if Model_restrictions[i]: del Model_Applied_Forces[i]
    
    return Model_Applied_Forces

def Calculate_Displacements(K_reduced, F_reduced):
    disp = np.linalg.inv(K_reduced) @ np.array(F_reduced)
    return disp

def ConstructDisp_full(Disp_reduced, Model_restrictions):
    Disp_full = []; j = 0;
    for i in range(len(Model_restrictions)):
        if Model_restrictions[i]:   Disp_full.append(0.0)
        else: Disp_full.append(Disp_reduced[j]); j += 1;
    
    return np.array(Disp_full)

def SetNodesDisp(Model_nodes, Model_Disp):
    node_DOF = 2; i = 0
    for node in Model_nodes:
        node.SetNodeDisp( (Model_Disp[i*node_DOF], Model_Disp[i*node_DOF+1]) )
        i += 1
        


    
# ************** MAIN FUNCTION **************

    # Specifying nodes
Model_nodes = []
Model_nodes.append( Node(1, (0.0, 0.0), (True, True), (0.0, 0.0)) )
Model_nodes.append( Node(2, (1.0, 0.0), (False, False), (0.0, -1000)))
Model_nodes.append( Node(3, (0.0, 1.0) , (True, False), (0.0, 0.0)) ) 
    
    # Specifying elements
Model_elements = []
Model_elements.append( Truss_elem(1, Model_nodes[1 -1], Model_nodes[2 -1], 70E9, 0.1) )
Model_elements.append( Truss_elem(2, Model_nodes[1 -1], Model_nodes[3 -1], 70E9, 0.1) )
Model_elements.append( Truss_elem(3, Model_nodes[2 -1], Model_nodes[3 -1], 70E9, 0.1) )

    # Assembling and reducing K, according to node DOF restrictions    
K_global = Assemble_K_global(Model_elements, np.shape(Model_nodes)[0])
Model_restrictions = GetModelRestrictions(Model_nodes)
K_reduced = Calculate_K_reduced(K_global, Model_restrictions)

    # Obtaining and reducing the vector of Applied Forces
Model_Applied_Forces = GetModelAppliedForces(Model_nodes)
Model_F_reduced = Calculate_F_reduced(Model_restrictions, Model_Applied_Forces)

    #Calculating the displacements of the Unrestricted DOFs
Disp_reduced = Calculate_Displacements(K_reduced, Model_F_reduced)
Model_Disp_Full = ConstructDisp_full(Disp_reduced, Model_restrictions)
SetNodesDisp(Model_nodes, Model_Disp_Full)

Elem_strains = []; Elem_stresses = []
for elem in Model_elements:
    Elem_strains.append(elem.CalculateElemStrain())
    Elem_stresses.append(elem.CalculateElemStress())
    
    
Elem2_strain = Model_elements[1].CalculateElemStrain()
Elem2_stress = Model_elements[1].CalculateElemStress()
