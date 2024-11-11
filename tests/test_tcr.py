###############################################################################
###################### Tests for the TCR Class ################################
###############################################################################


from src.TCR import *
from src.pMHC import *
import pytest 


# Test class for TCR
class TestTCR:

    def test_valid_minimal_initialization(self):
        # Test for successful initialization with valid data from GILGFVFTL_A*02:01
        tcr = TCR(cdr3b='CASSIRSSYEQYF', trbv='TRBV19*01', trbj='TRBJ2-7*01')
        assert tcr.cdr3b == "CASSIRSSYEQYF"
        assert tcr.trbv == "TRBV19*01"
        assert tcr.trbj == "TRBJ2-7*01"
        assert tcr.get_pMHCs() == set()
        assert tcr.get_references() == set()

    def test_initialization_with_optional_fields(self):
        # Test initialization with some optional fields
        tcr = TCR(cdr3b='CASSIRSSYEQYF', trbv='TRBV19*01', trbj='TRBJ2-7*01',
                  cdr3a="CATGLTGGGNKLTF", trav="TRAV17*01", traj="TRAJ10*01")

        assert tcr.cdr3a == "CATGLTGGGNKLTF"
        assert tcr.trav == "TRAV17*01"
        assert tcr.traj == "TRAJ10*01"

    def test_invalid_initialization(self):
        # Test initialization with invalid/mandatory fields
        with pytest.raises(ValueError):
            TCR(cdr3b="SADAF", trbv="TBRFV", trbj="TRBJ42069")
    
    def test_incorrect_IMGT_formatting(self):
        tcr = TCR(cdr3b='cASsIRSsYEqYF', trbv='TRBV19*1', trbj='TRBJ2-07*1',
                  cdr3a="CATGLTGGGNKLTF", trav="TRAV17*1", traj="TRAJ10*1")
        
        assert tcr.cdr3b == "CASSIRSSYEQYF"
        assert tcr.trbv == "TRBV19*01"
        assert tcr.trbj == "TRBJ2-7*01"
        assert tcr.cdr3a == "CATGLTGGGNKLTF"
        assert tcr.trav == "TRAV17*01"
        assert tcr.traj == "TRAJ10*01"
    
    def test_repr(self):
        # Test repr
        tcr = TCR(cdr3b='CASSIRSSYEQYF', trbv='TRBV19*01', trbj='TRBJ2-7*01')
        assert eval(repr(tcr)) == tcr

    def test_str(self):
        # Test str
        tcr = TCR(cdr3b='CASSIRSSYEQYF', trbv='TRBV19*01', trbj='TRBJ2-7*01', use_cdr3b=True)
        assert str(tcr) == "CASSIRSSYEQYF"
        

    def test_data_integrity(self):
        # Test to see if the frozen=True holds
        tcr = TCR(cdr3b='CASSIRSSYEQYF', trbv='TRBV19*01', trbj='TRBJ2-7*01',
                  cdr3a="CATGLTGGGNKLTF", trav="TRAV17*01", traj="TRAJ10*01")
        
        with pytest.raises(AttributeError):
            tcr.cdr3b = "SOME FAKE VALUE"
            tcr.trbv = "A FAKE GENE"
            tcr.trbj = "NUTM1?"
        
        assert tcr.cdr3a == "CATGLTGGGNKLTF"
        assert tcr.trav == "TRAV17*01"
        assert tcr.traj == "TRAJ10*01"
        
    def test_add_pMHC(self):
        # Test adding a pMHC
        tcr = TCR(cdr3b='cASsIRSsYEqYF', trbv='TRBV19*1', trbj='TRBJ2-07*1',
                  cdr3a="CATGLTGGGNKLTF", trav="TRAV17*1", traj="TRAJ10*1")
        pmhc = pMHC("GILGFVFTL", "A*02:01")
        tcr.add_pMHC(pmhc)
        assert pmhc in tcr.pMHCs

    def test_get_pMHCs(self):
        # Test get_pMHCs method
        pmhc1 = pMHC("GILGFVFTL", "A*02:01")
        pmhc2 = pMHC("GLCTLVAML", "B*07:02")
        tcr = TCR(cdr3b='cASsIRSsYEqYF', trbv='TRBV19*1', trbj='TRBJ2-07*1', pMHC=pmhc1)
        tcr.add_pMHC(pmhc2)
        assert tcr.get_pMHCs() == {pmhc1, pmhc2}

    def test_add_reference(self):
        # Test adding a reference
        tcr = TCR(cdr3b='cASsIRSsYEqYF', trbv='TRBV19*1', trbj='TRBJ2-07*1',
                  cdr3a="CATGLTGGGNKLTF", trav="TRAV17*1", traj="TRAJ10*1")
        tcr.add_reference("Ref1")
        assert "Ref1" in tcr.get_references()
        assert len(tcr.get_references())==1
        tcr.add_reference("Ref2")
        tcr.add_reference("Ref3")
        assert "Ref2" in tcr.get_references()
        assert "Ref3" in tcr.get_references()
        assert len(tcr.get_references())==3
            
    def test_get_references(self):
        # Test get_references method
        tcr = TCR(cdr3b='cASsIRSsYEqYF', trbv='TRBV19*1', trbj='TRBJ2-07*1',
                  cdr3a="CATGLTGGGNKLTF", trav="TRAV17*1", traj="TRAJ10*1",
                  reference="Ref1")
        assert tcr.get_references() == {"Ref1"}
        
        