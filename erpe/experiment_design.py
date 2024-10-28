# typing import
from typing import List, Dict, Iterable
from pygsti.circuits import Circuit
import pygsti
import matplotlib.pyplot as plt
import numpy as np
from abc import ABC, abstractmethod

class ExperimentDesignBase(ABC):
    def __init__(self, 
                    depths : List[int],
                    line_labels : List[str],
                    check_definition : bool = True
                    ):
        """Base class for an experiment design.

        Args:
            germs (List[str]): list of germs.
            inphase_state_preps (List[str]): list of in-phase state preps.
            quadruature_state_preps (List[str]): list of quadrature state preps.
            meas_fiducial (List[str]): list of measurement fiducials.
            depths (List[int]): list of depths.
        """
        self.depths = depths
        self.line_labels = line_labels

        # # self.quadruature_state_preps = quadruature_state_preps
        # # self.meas_fiducials = meas_fiducials
        # self.depths = depths
        # self.line_labels = line_labels
        # self.signal_subpace_labels = signal_subpace_labels
        # # self.cos_type_plus_outcome_labels = cos_type_plus_outcome_labels
        # # self.cos_type_minus_outcome_labels = cos_type_minus_outcome_labels
        # # self.sin_type_plus_outcome_labels = sin_type_plus_outcome_labels
        # # self.sin_type_minus_outcome_labels = sin_type_minus_outcome_labels
        self.circuit_dict = self._construct_edesign()

    def __str__(self) -> str:
        return f"Experiment design with {len(self.germs)} germs and {len(self.depths)} depths"
    
    
    # def germ_to_hash(self, germ):
    #     return '.'.join([g[0] for g in germ])
    
    # def germ_hash(self, idx):
    #     return self.germ_to_hash(self.germs[idx])

    @property
    #@abstractmethod
    def germs(self):
        pass
    
    @property
    #@abstractmethod
    def preparation_fiducials(self): 
        pass

    @property
    #@abstractmethod
    def measurement_fiducials(self):
        pass

    @property
    #@abstractmethod
    def signal_subspaces(self):
        pass
    
    # @property
    # def num_circuits(self) -> int:
    #     """Number of circuits in the experiment design.

    #     Returns:
    #         int: number of circuits.
    #     """
    #     return 2 * len(self.germs) * len(self.depths)
    
    @property
    def circuit_list(self) -> List[Circuit]:
        """List of circuits in the experiment design.

        Returns:
            List[Circuit]: list of circuits.
        """
        all_circs = []
        for germ in self.circuit_dict.keys():
            for measurement in self.circuit_dict[germ].keys():
                for depth in self.circuit_dict[germ][measurement]['I'].keys():
                    all_circs.append(self.circuit_dict[germ][measurement]['I'][depth])
                    all_circs.append(self.circuit_dict[germ][measurement]['Q'][depth])
        return pygsti.remove_duplicates(all_circs)
        
    def _construct_edesign(self):
        """Construct the experiment design.
        Returns:
            Dict: experiment design.
        """

        edesign = {}
        for germ in self.preparation_fiducials.keys():
            edesign[germ] = {}
            for measurement in self.measurement_fiducials[germ].keys():
                edesign[germ][measurement] = {'I' : {}, 'Q' : {}}
                for depth in self.depths:
                    edesign[germ][measurement]['I'][depth] = Circuit(
                        self.preparation_fiducials[germ][measurement]['I'] 
                        + germ*depth 
                        + self.measurement_fiducials[germ][measurement], 
                        line_labels=list(self.qids)
                    )
                    edesign[germ][measurement]['Q'][depth] = Circuit(
                        self.preparation_fiducials[germ][measurement]['Q'] 
                        + germ*depth 
                        + self.measurement_fiducials[germ][measurement], 
                        line_labels=list(self.qids)
                    )
        return edesign
    
    def make_dataset(self, model, num_shots_per_circuit):
        """
        Make a dataset from the experiment design.
        """
        ds = pygsti.data.simulate_data(model, self.circuit_list, num_samples=num_shots_per_circuit)
        return ds
        
class EDesign_1QXI(ExperimentDesignBase):
    def __init__(self, 
                 depths : List[int],
                 qids : str='Q0'):
        depths = depths
        line_labels = qids
        assert len(qids) == 1, "Only one qubit is allowed for this experiment design."
        self.qids = qids
        super().__init__(depths, line_labels)

    @property
    def germs(self):
        gxpi2 = Circuit([('Gxpi2', self.qids[0])], line_labels=self.qids)
        gzpi2 = Circuit([('Gzpi2', self.qids[0])], line_labels=self.qids)   
        gi = Circuit([('Gi', self.qids[0])], line_labels=self.qids)

        germ1 = gxpi2
        germ2 = gzpi2 + gxpi2*2 + gzpi2*2 + gxpi2*2 + gzpi2
        germ3 = gi
        return [germ1, germ2, germ3]
    
    # def inphase_prep_fid_dict(self): 
    #     gxpi2 = [('Gxpi2', self.qid)]
    #     gzpi2 = [('Gzpi2', self.qid)]
    #     gypi2 = gzpi2 + gxpi2 + gzpi2*3
    #     return {
    #         Circuit(self.germs[0]) : [], 
    #         Circuit(self.germs[1]) : [], 
    #         Circuit(self.germs[2]) : gypi2
    #     }
    
    # def quadrature_prep_fid_dict(self):
    #     gxpi2 = [('Gxpi2', self.qid)]
    #     return {
    #         Circuit(self.germs[0]) : gxpi2, 
    #         Circuit(self.germs[1]) : gxpi2, 
    #         Circuit(self.germs[2]) : gxpi2
    #     }

    @property
    def preparation_fiducials(self): 
        """
        Define the preparation fiducials for the experiment design.

        Since this is a single-qubit edesign, each germ has a single measurement type.        
        """
        gxpi2 = [('Gxpi2', self.qids[0])]
        gzpi2 = [('Gzpi2', self.qids[0])]
        gypi2 = gzpi2 + gxpi2 + gzpi2*3
        return {
            Circuit(self.germs[0]): {
                '0' : {
                    'I': Circuit([], line_labels=self.qids),
                    'Q': Circuit([('Gxpi2', self.qids[0])], line_labels=self.qids),
                },
            }, 
            Circuit(self.germs[1]): {
                '0' : {
                    'I': Circuit([], line_labels=self.qids),
                    'Q': Circuit([('Gxpi2', self.qids[0])], line_labels=self.qids),
                },
            },
            Circuit(self.germs[2]): {
                '+' : {
                    'I': Circuit(gypi2,  line_labels=self.qids),
                    'Q': Circuit([('Gxpi2', self.qids[0])],  line_labels=self.qids),
                },
            }
        }
    
    @property
    def measurement_fiducials(self):
        gxpi2 = [('Gxpi2', self.qids[0])]
        gzpi2 = [('Gzpi2', self.qids[0])]
        gypi2 = gzpi2 + gxpi2 + gzpi2*3
        return {
            Circuit(self.germs[0]): {
                '0' : Circuit([], line_labels=self.qids),
            }, 
            Circuit(self.germs[1]): {
                '0' : Circuit([], line_labels=self.qids),
            }, 
            Circuit(self.germs[2]): {
                '+' : Circuit(gypi2*3, line_labels=self.qids),
            }, 
        }
    
    # def measurment_fid_dict(self):
    #     gxpi2 = [('Gxpi2', self.qid)]
    #     gzpi2 = [('Gzpi2', self.qid)]
    #     gypi2 = gzpi2 + gxpi2 + gzpi2*3
    #     return {
    #         Circuit(self.germs[0]) : [], 
    #         Circuit(self.germs[1]) : [], 
    #         Circuit(self.germs[2]) : gypi2*3
    #     }
    
    # def signal_subspace_dict(self):
        # return {
        #     Circuit(self.germs[0]) : {
        #         'I' : {'+' : ['0', ], '-' : ['1', ]},
        #         'Q' : {'+' : ['0', ], '-' : ['1', ]}
        #     },
        #     Circuit(self.germs[1]) : {
        #         'I' : {'+' : ['0', ], '-' : ['1', ]},
        #         'Q' : {'+' : ['0', ], '-' : ['1', ]}
        #     },
        #     Circuit(self.germs[2]) : {
        #         'I' : {'+' : ['0', ], '-' : ['1', ]},
        #         'Q' : {'+' : ['0', ], '-' : ['1', ]}
        #     }
        # }

    @property
    def signal_subspaces(self):
        return {
            Circuit(self.germs[0]) : {
                '0' : {
                    'I' : {'+' : ['0', ], '-' : ['1', ]},
                    'Q' : {'+' : ['0', ], '-' : ['1', ]}
                }
            },
            Circuit(self.germs[1]) : {
                '0' : {
                    'I' : {'+' : ['0', ], '-' : ['1', ]},
                    'Q' : {'+' : ['0', ], '-' : ['1', ]}
                }
            },
            Circuit(self.germs[2]) : {
                '+' : {
                    'I' : {'+' : ['0', ], '-' : ['1', ]},
                    'Q' : {'+' : ['0', ], '-' : ['1', ]}
                }
            }
        }
        # return {
        #     (('Gcz', *self.qids)): {
        #         '0+' : {
        #             'I': {'+' : ['00', ], '-' : ['01', ]},
        #             'Q' : {'+' : ['00', ], '-' : ['01', ]}
        #         },
        #         '1+' : {
        #             'I': {'+' : ['10', ], '-' : ['11', ]},
        #             'Q' : {'+' : ['10', ], '-' : ['11', ]}
        #         },
        #         '+1' : {
        #             'I': {'+' : ['01', ], '-' : ['11', ]}, line_labels=self.qids),
        #             'Q' : {'+' : ['01', ], '-' : ['11', ]}
        #         }
        #     }
        # }
    

class EDesign_CZ(ExperimentDesignBase):
    """
    TODO: adapt the base class to accomodate the CZ experiment design.
    """
    def __init__(self, 
                 depths : List[int],
                 qids):
        self.depths = depths
        assert len(qids) == 2, "Only two qubits are allowed for this experiment design."
        self.qids = qids
        super().__init__(depths, line_labels=qids)

    @property
    def germs(self):
        gcz = [('Gcz', *self.qids), ]
        germ1 = gcz
        return [Circuit(germ1, line_labels=self.qids)]
    
    @property
    def preparation_fiducials(self): 
        return {
            self.germs[0]: {
                '0+' : {
                    'I': Circuit([('Gypi2', self.qids[1])], line_labels=self.qids),
                    'Q': Circuit([('Gxpi2', self.qids[1])], line_labels=self.qids),
                }, 
                '1+' : {
                    'I': Circuit([('Gypi2', self.qids[1]), 
                                  ('Gxpi2', self.qids[0]), 
                                  ('Gxpi2', self.qids[0]), ], line_labels=self.qids),
                    'Q': Circuit([('Gxpi2', self.qids[1]), 
                                  ('Gxpi2', self.qids[0]), 
                                  ('Gxpi2', self.qids[0]), ], line_labels=self.qids),
                },
                '+1' : {
                    'I': Circuit([('Gypi2', self.qids[0]), 
                                  ('Gxpi2', self.qids[1]), 
                                  ('Gxpi2', self.qids[1]), ], line_labels=self.qids),
                    'Q': Circuit([('Gxpi2', self.qids[0]),
                                  ('Gxpi2', self.qids[1]), 
                                  ('Gxpi2', self.qids[1]), ], line_labels=self.qids),
                }
            }
        }
    
    @property
    def measurement_fiducials(self):
        return {
            self.germs[0]: {
                '0+' : Circuit([('Gypi2', self.qids[1])]*3, line_labels=self.qids),
                '1+' : Circuit([('Gypi2', self.qids[1])]*3, line_labels=self.qids),
                '+1' : Circuit([('Gypi2', self.qids[0])]*3, line_labels=self.qids),
            }
        }
    
    @property
    def signal_subspaces(self):
        return {
            self.germs[0]: {
                '0+' : {
                    'I': {'+' : ['00', ], '-' : ['01', ]},
                    'Q' : {'+' : ['00', ], '-' : ['01', ]}
                },
                '1+' : {
                    'I': {'+' : ['10', ], '-' : ['11', ]},
                    'Q' : {'+' : ['10', ], '-' : ['11', ]}
                },
                '+1' : {
                    'I': {'+' : ['01', ], '-' : ['11', ]},
                    'Q' : {'+' : ['01', ], '-' : ['11', ]}
                }

            }
            # Circuit(self.germs[0]) : {
            #     'I' : {'+' : ['0', ], '-' : ['1', ]},
            #     'Q' : {'+' : ['0', ], '-' : ['1', ]}
            # },
            # Circuit(self.germs[1]) : {
            #     'I' : {'+' : ['0', ], '-' : ['1', ]},
            #     'Q' : {'+' : ['0', ], '-' : ['1', ]}
            # },
            # Circuit(self.germs[2]) : {
            #     'I' : {'+' : ['0', ], '-' : ['1', ]},
            #     'Q' : {'+' : ['0', ], '-' : ['1', ]}
            # }
        }
    
    # @property
    # def circuit_list(self) -> List[Circuit]:
    #     """List of circuits in the experiment design.

    #     Returns:
    #         List[Circuit]: list of circuits.
    #     """
    #     all_circs = []
    #     for germ in self.circuit_dict.keys():
    #         for measurement in self.circuit_dict[germ].keys():
    #             for depth in self.circuit_dict[germ][measurement]['I'].keys():
    #                 all_circs.append(self.circuit_dict[germ][measurement]['I'][depth])
    #                 all_circs.append(self.circuit_dict[germ][measurement]['Q'][depth])
    #     return pygsti.remove_duplicates(all_circs)
        
    # def _construct_edesign(self):
    #     """Construct the experiment design.
    #     Returns:
    #         Dict: experiment design.
    #     """

    #     edesign = {}
    #     for germ in self.preparation_fiducials.keys():
    #         edesign[germ] = {}
    #         for measurement in self.measurement_fiducials[germ].keys():
    #             edesign[germ][measurement] = {'I' : {}, 'Q' : {}}
    #             for depth in self.depths:
    #                 edesign[germ][measurement]['I'][depth] = Circuit(
    #                     self.preparation_fiducials[germ][measurement]['I'] 
    #                     + [germ]*depth 
    #                     + self.measurement_fiducials[germ][measurement], 
    #                     line_labels=list(self.qids)
    #                 )
    #                 edesign[germ][measurement]['Q'][depth] = Circuit(
    #                     self.preparation_fiducials[germ][measurement]['Q'] 
    #                     + [germ]*depth 
    #                     + self.measurement_fiducials[germ][measurement], 
    #                     line_labels=list(self.qids)
    #                 )
    #     return edesign
    
    # def make_dataset(self, model, num_shots_per_circuit):
    #     """
    #     Make a dataset from the experiment design.
    #     """
    #     ds = pygsti.data.simulate_data(model, self.circuit_list, num_samples=num_shots_per_circuit)
    #     return ds

# class EDesign_1QXZ(ExperimentDesign):
#     def __init__(self, 
#                  depths : List[int],
#                  qid : str='Q0'):
#         depths = depths
#         line_labels = [qid]
#         self.qid = qid

#         germs = self.list_1qxz_germs()
#         inphase_state_preps = self.list_1qxz_inphase_prep_fids()
#         quadrature_state_preps = self.list_1qxz_quadrature_prep_fids()
#         meas_fiducials = self.list_1qxz_measurment_fids()
#         germ_quadrature_labels_Q1 = {
#         'X': {
#             '+': ['00'],
#             '-': ['01']
#         },
#         'I' : {
#             '+' : ['00'], 
#             '-' : ['02']
#         },
#         'XZZXZZ': {
#             '+': ['01'],
#             '-': ['02']
#         },}
#         super().__init__(germs,inphase_state_preps, quadrature_state_preps, meas_fiducials, 
#                          cos_type_plus_outcome_labels, cos_type_minus_outcome_labels, sin_type_plus_outcome_labels, 
#                          sin_type_minus_outcome_labels, depths, line_labels)
        

#     def list_1qxz_germs(self):
#         gxpi2 = [('Gxpi2', self.qid)]
#         gzpi2 = [('Gzpi2', self.qid)]

#         germ1 = gxpi2
#         germ2 = gxpi2 + gzpi2 + gzpi2 + gxpi2 + gzpi2 + gzpi2
#         germ3 = gxpi2*2 + gzpi2*2
#         return [germ1, germ2, germ3]
    
#     def list_1qxz_inphase_prep_fids(self):
#         gxpi2 = [('Gxpi2', self.qid)]
#         gzpi2 = [('Gzpi2', self.qid)]
#         gypi2 = gzpi2 + gxpi2 + gzpi2*3
#         return [[], gypi2, []]
    
#     def list_1qxz_quadrature_prep_fids(self):
#         gxpi2 = [('Gxpi2', self.qid)]
#         gzpi2 = [('Gzpi2', self.qid)]
#         gypi2 = gzpi2 + gxpi2 + gzpi2*3
#         return [gxpi2, gxpi2, gypi2]
    
#     def list_1qxz_measurment_fids(self):
#         gxpi2 = [('Gxpi2', self.qid)]
#         gzpi2 = [('Gzpi2', self.qid)]
#         gypi2 = gzpi2 + gxpi2 + gzpi2*3
#         return [[], gypi2*3, []]


    
# class EDesign_1QXI(ExperimentDesign):
#     def __init__(self, 
#                  depths : List[int],
#                  qid : str='Q0'):
#         depths = depths
#         line_labels = [qid]
#         self.qid = qid

#         germs = self.list_1qXI_germs()
#         inphase_state_preps = self.list_1qXI_inphase_prep_fids()
#         quadruature_state_preps = self.list_1qXI_quadrature_prep_fids()
#         meas_fiducials = self.list_1qXI_measurment_fids()
#         # cos_type_plus_outcome_labels = [('1', ), ('1', ), ('1', )]
#         # cos_type_minus_outcome_labels = [('0', ), ('0', ), ('0', )]
#         # sin_type_plus_outcome_labels = [('0', ), ('0', ), ('0', )]
#         # sin_type_minus_outcome_labels = [('1', ), ('1', ), ('1', )]
#         super().__init__(germs,inphase_state_preps, quadruature_state_preps, meas_fiducials, 
#                          cos_type_plus_outcome_labels, cos_type_minus_outcome_labels, sin_type_plus_outcome_labels, 
#                          sin_type_minus_outcome_labels, depths, line_labels)
        

#     def list_1qXI_germs(self):
#         gxpi2 = [('Gxpi2', self.qid)]
#         gzpi2 = [('Gzpi2', self.qid)]
#         gi = [('Gi', self.qid)]

#         germ1 = gxpi2
#         germ2 = gxpi2 + gzpi2 + gzpi2 + gxpi2 + gzpi2 + gzpi2
#         germ3 = gi
#         return [germ1, germ2, germ3]
    
#     def list_1qXI_inphase_prep_fids(self): 
#         gxpi2 = [('Gxpi2', self.qid)]
#         gzpi2 = [('Gzpi2', self.qid)]
#         gypi2 = gzpi2 + gxpi2 + gzpi2*3
#         return [[], gypi2, gypi2]
    
#     def list_1qXI_quadrature_prep_fids(self):
#         gxpi2 = [('Gxpi2', self.qid)]
#         gzpi2 = [('Gzpi2', self.qid)]
#         gypi2 = gzpi2 + gxpi2 + gzpi2*3
#         return [gxpi2, gxpi2, gxpi2]
    
#     def list_1qXI_measurment_fids(self):
#         gxpi2 = [('Gxpi2', self.qid)]
#         gzpi2 = [('Gzpi2', self.qid)]
#         gypi2 = gzpi2 + gxpi2 + gzpi2*3
#         return [[], gypi2*3, gypi2*3]
        
#     def plot_dataset(self, ds, target_model=None):
#         """Make a bunch of subplots for the dataset."""

#         for germ in self.germs:
#             germ_str = self.germ_to_hash(germ)
#             fig, ax = plt.subplots(2, len(self.depths), figsize=(10, 5), sharey=True)
#             # set the germ as the super title
#             fig.suptitle(germ_str)
#             for idx, depth in enumerate(self.depths):
#                 inphase_circ  = self.circuit_dict[germ_str]['I'][depth]
#                 quad_circ = self.circuit_dict[germ_str]['Q'][depth]
#                 inphase_data = ds[inphase_circ].counts
#                 quad_data = ds[quad_circ].counts
#                 if target_model is not None:
#                     inphase_target_probs = target_model.probabilities(inphase_circ)
#                     quad_target_probs = target_model.probabilities(quad_circ)
#                     num_counts = sum(inphase_data.values())

#                 ax[0, idx].bar([0, 1], [inphase_data['0'], inphase_data['1']])
#                 if target_model is not None:
#                     ax[0, idx].bar([0, 1], [inphase_target_probs['0']*num_counts, inphase_target_probs['1']*num_counts], alpha=0.5)
#                 ax[0, idx].set_title(f'I at {depth}')
#                 ax[1, idx].bar([0, 1], [quad_data['0'], quad_data['1']])
#                 if target_model is not None:
#                     ax[1, idx].bar([0, 1], [quad_target_probs['0']*num_counts, quad_target_probs['1']*num_counts], alpha=0.5)
#                 ax[1, idx].set_title(f'Q at {depth}')
#             # set share axis

#             plt.tight_layout()
