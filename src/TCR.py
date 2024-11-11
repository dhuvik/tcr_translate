"""
The purpose of this python3 script is to implement the TCR dataclass.
"""

# Import Dependencies
from functools import cached_property
from dataclasses import dataclass, field
from typing import Optional, Set
from src.constants import *
from src.pMHC import pMHC
import tidytcells as tdtc


@dataclass(frozen=True)
class TCR:
    """
    # TODO: Implement stitchr so that full TRA and TRB can be imputed if possible
    
    Define a Meaningful TCR Object that is used to store TCR information in the
    context of paired immmunogenicity data. Standardization is performed using the
    'tidytcells' package authored by Yuta Nagano.

    Attributes:
        
        cdr3b (str, required): The CDR3 beta sequence. 
        trbv (str, required): The TRBV gene in IMGT format.
        trbj (str, required): The TRBJ gene in IMGT format
        trbd (str, optional): The TRBD gene in IMGT format. Defaults to None.
        tcrb_full (str, optional): The full beta TCR sequence. Defaults to None.
        cdr3a (str, optional): The CDR3 alpha sequence. Defaults to None.
        trav (str, optional): The TRAV gene in IMGT format. Defaults to None.
        traj (str, optional): The TRAJ gene in IMGT format. Defaults to None.
        trad (str, optional): The TRAD gene in IMGT format. Defaults to None.
        tcra_full (str, optional): The full alpha TCR sequence. Defaults to None.
        pMHC (pMHC, optional): The pMHC object recognized by this TCR. Defaults to None.
        reference (str, optional): The reference for this TCR. Defaults to None.

    Implements:
        __init__: Initialize the TCR object
        __post_init__: Standardize the input fields using IMGT format and check for
            minimum required inputs.
        __repr__: Return a uniquely identifiable representation of the TCR object.
        __str__: Return an string representation of the TCR object for user interaction.
        __eq__: Check if two TCR objects are functionally equivalent. As such it does not 
            check for equality across all the possible fields of the TCR but checks
            for the CDR3 regions as well as the TRBV and TRBJ genes.
        __hash__: Hash the TCR object
        add_pMHC: Add a pMHC to the set of cognate pMHCs recognized by this TCR
        add_reference: Add a reference to the set of references for this TCR
        get_pMHCs: Get the set of pMHCs recognized by this TCR
        get_references: Get the set of references for this TCR
    
    Returns:
        TCR Object: A TCR dataclass object that can be used to store TCR information    
    """

    # TRB Information (Immutable)
    cdr3b: str
    trbv: str
    trbj: str
    trbd: Optional[str] = None
    tcrb_full: Optional[str] = None
    
    # TRA Infromation (Immutable)
    cdr3a: Optional[str] = None
    trav: Optional[str] = None
    traj: Optional[str] = None
    trad: Optional[str] = None
    tcra_full: Optional[str] = None

    # Dataset-Level Information
    use_cdr3b: bool = True

    
    # Extrinsic Information (Immutable items to be added to mutable collection)
    pMHC: Optional[object] = None  # Assuming pMHC is an object type
    reference: Optional[str] = None

    # Extrinsic Information (Mutable Collection)
    pMHCs: Set[pMHC] = field(default_factory=set, compare=False, hash=False)
    references: Set[str] = field(default_factory=set, compare=False, hash=False)

    def __post_init__(self):
        """
        Go through and make sure that fields that are not None are converted parsed
        and formatted using IMGT format. S/o Yuta Nagano.
        """
        
        # Retain required info for bookkeeping
        input_cdr3b = self.cdr3b
        input_trbv = self.trbv
        input_trbj = self.trbj
        
        # Format the TRB information
        object.__setattr__(self, 'cdr3b', tdtc.junction.standardize(seq=self.cdr3b, strict=False, suppress_warnings=False))
        object.__setattr__(self, 'trbv', tdtc.tr.standardize(gene=self.trbv, precision='allele', suppress_warnings=False))
        object.__setattr__(self, 'trbj', tdtc.tr.standardize(gene=self.trbj, precision='allele', suppress_warnings=False))
        if self.trbd is not None:
            object.__setattr__(self, 'trbd', tdtc.tr.standardize(gene=self.trbd, precision='allele', suppress_warnings=True))

        # Format the TRA information
        if self.cdr3a is not None:
            object.__setattr__(self, 'cdr3a', tdtc.junction.standardize(seq=self.cdr3a, strict=False, suppress_warnings=False))
        if self.trav is not None:
            object.__setattr__(self, 'trav', tdtc.tr.standardize(gene=self.trav, precision='allele', suppress_warnings=True))
        if self.traj is not None:
            object.__setattr__(self, 'traj', tdtc.tr.standardize(gene=self.traj, precision='allele', suppress_warnings=True))
        if self.trad is not None:
            object.__setattr__(self, 'trad', tdtc.tr.standardize(gene=self.trad, precision='allele', suppress_warnings=True))

        # Add extrinsic information to mutable collection
        if self.pMHC is not None:
            self.add_pMHC(self.pMHC)
            
        if self.reference is not None:
            self.add_reference(self.reference)
                
        # Check validity of required input information
        if self.cdr3b is None or self.trbv is None or self.trbj is None:
            raise ValueError(f"You must supply a valid CDR3b, TRBV, and TRBJ. Got CDR3b:{input_cdr3b}, TRBV:{input_trbv}, TRBJ:{input_trbj}")
        else:
            pass
            
    def __repr__(self):
        """
        Returns the representation of the TCR obeject used for tokenization.
        Currently only handles TCRb information.
        """
        return f'''TCR(cdr3a="{self.cdr3a}", cdr3b="{self.cdr3b}",
                trav="{self.trav}", trbv="{self.trbv}", 
                traj="{self.traj}", trbj="{self.trbj}",
                trad="{self.trad}", trbd="{self.trbd}",
                tcra_full="{self.tcra_full}", tcrb_full="{self.tcrb_full}",
                reference={self.get_references()}, use_cdr3b={self.use_cdr3b})'''

            
    def __str__(self):
        """
        Return an string representation of the TCR object for user interaction.
        """
        if self.use_cdr3b:
            return f'{self.cdr3b}'
        else:
            return f'{self.tcrb_full}'
        

    def __eq__(self, other):
        """
        Check if two TCR objects are functionally equivalent. As such it does not 
        check for equality across all the possible fields of the TCR but checks
        for 
        """
        if not isinstance(other, TCR):
            return NotImplemented
        
        return self.cdr3b == other.cdr3b and self.trbv == other.trbv and self.trbj == other.trbj and self.cdr3a == other.cdr3a
    
    def __hash__(self):
        """
        Hash the TCR object
        """
        return hash((self.cdr3b, self.trbv, self.trbj, self.cdr3a))
    
    def add_pMHC(self, pMHC):
        """
        Add a pMHC to the set of cognate pMHCs recognized by this TCR

        Args:
            pMHC (pMHC Object): The pMHC to add.
        """
        self.pMHCs.add(pMHC)

    def add_reference(self, reference):
        """
        Add a reference to the set of references for this TCR

        Args:
            reference (str): The reference to add
        """
        if isinstance(reference, str):
            self.references.add(reference)
        elif isinstance(reference, set) or isinstance(reference, list):
            self.references.update(reference)
        else:
            pass

    def get_pMHCs(self):
        """
        Get the set of pMHCs recognized by this TCR

        Returns:
            set: The set of pMHCs recognized by this TCR
        """
        return self.pMHCs
    
    def get_references(self):
        """
        Get the set of references for this TCR

        Returns:
            set: The set of references for this TCR
        """
        return self.references
    