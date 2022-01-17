# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 20:16:17 2022

@author: manud
"""

from Truss_simple_structure import *




# ************** MAIN FUNCTION **************

    # Specifying nodes and Model
Model_chapt3_ex = Model()

Model_chapt3_ex.AddNode( Node(1, (0.0, 0.0), (True, True), (0.0, 0.0)) )
Model_chapt3_ex.AddNode( Node(2, (1.0, 0.0), (False, False), (0.0, -1000)))
Model_chapt3_ex.AddNode( Node(3, (0.0, 1.0) , (True, False), (0.0, 0.0)) )
Model_chapt3_ex.AddNode( Node(4, (2.0, 1.0) , (False, False), (5000, 0.0)) )
    
    # Specifying elements
Model_nodes = Model_chapt3_ex.GetModelNodes()
Model_chapt3_ex.AddElement( Truss_elem(1, Model_nodes[1 - 1], Model_nodes[2 -1 ], 70E9, 0.1) )
Model_chapt3_ex.AddElement( Truss_elem(2, Model_nodes[1 - 1], Model_nodes[3 -1 ], 70E9, 0.1) )
Model_chapt3_ex.AddElement( Truss_elem(3, Model_nodes[2 - 1], Model_nodes[3 -1 ], 70E9, 0.1) )
Model_chapt3_ex.AddElement( Truss_elem(4, Model_nodes[3 - 1], Model_nodes[4 -1 ], 70E9, 0.1) )
Model_chapt3_ex.AddElement( Truss_elem(5, Model_nodes[2 - 1], Model_nodes[4 -1 ], 70E9, 0.1) )




# Performing calculations

(   K_global, Model_restrictions, K_reduced, 
    Model_Applied_Forces, Model_F_reduced, 
    Disp_reduced, Model_Disp_Full, 
    Model_strains, Model_stresses
    ) = PerformComputations(Model_chapt3_ex)


# Displaying nodes
    
    
Model_chapt3_ex.PlotNodes()
Model_chapt3_ex.PlotElements()
Model_chapt3_ex.PlotRestrictions()
Model_chapt3_ex.PlotForces()

plt.margins(0.2)

Model_chapt3_ex.PlotStructureDisp_Exagerated(100000)

plt.grid()
plt.legend()
plt.margins(0.2)
plt.tight_layout()
plt.show()