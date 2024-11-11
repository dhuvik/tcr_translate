###############################################################################
###################### Tests for the pMHC Class ###############################
###############################################################################

from src.TCR import *
from src.pMHC import *
import pytest 


# Test class for pMHC
class TestpMHC:

    def test_valid_minimal_initialization(self):
        # Test for successful initialization with valid data 
        pmhc = pMHC(peptide='GILGFVFTL', hla_allele='HLA-A*02:01')
        assert pmhc.peptide == "GILGFVFTL"
        assert pmhc.hla_allele == "HLA-A*02:01"
        assert pmhc.allele == "HLA-A*02:01"
        assert pmhc.pseudo == "YFAMYGEKVAHTHVDTLYVRYHYYTWAVLAYTWY"
        assert pmhc.mhc == ("MAVMAPRTLVLLLSGALALTQTWAGSHSMRYFFTSVSRPGRGEPRFIAV"  
                            "GYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDGETRKVKAHSQTH"
                            "RVDLGTLRGYYNQSEAGSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDY"
                            "IALKEDLRSWTAADMAAQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLE"
                            "NGKETLQRTDAPKTHMTHHAVSDHEATLRCWALSFYPAEITLTWQRDGED"
                            "QTQDTELVETRPAGDGTFQKWAAVVVPSGQEQRYTCHVQHEGLPKPLTLRW"
                            "EPSSQPTIPIVGIIAGLVLFGAVITGAVVAAVMWRRKSSDRKGGSYSQAAS"
                            "SDSAQGSDVSLTACKV") 
        assert pmhc.tcrs == set()
        assert pmhc.references == set()

    def test_invalid_initialization(self):
        # Test initialization with invalid/mandatory fields
        with pytest.raises(ValueError):
            pmhc = pMHC(peptide="GILGFVFTL", hla_allele="HLA-B2")
            print(pmhc.allele)
            assert pmhc.hla_allele == "HLA-B2"
        

    def test_tidy_tcell_impute(self):
        # Test initialization with invalid but recoverable HLA fields
        pmhc = pMHC(peptide="GilgFVfTL", hla_allele="HLA-A0101", eager_impute=False)
        assert pmhc.peptide == "GILGFVFTL"
        assert pmhc.hla_allele == "HLA-A0101"
        assert pmhc.allele == "HLA-A*01:01" 

        pmhc = pMHC(peptide="GilgFVfTL", hla_allele="HLA-A0101", eager_impute=False)
        assert pmhc.peptide == "GILGFVFTL"
        assert pmhc.hla_allele == "HLA-A0101"
        assert pmhc.allele == "HLA-A*01:01" 
    
    def test_eager_impute(self):
        # Test initialization with invalid/mandatory fields
        pmhc1 = pMHC(peptide="GILGFVFTL", hla_allele="HLA-A2", eager_impute=True)
        assert pmhc1.hla_allele == "HLA-A2"
        assert pmhc1.allele == "HLA-A*02:01"
        
        pmhc2 = pMHC(peptide="GILGFVFTL", hla_allele="HLA-A0101", eager_impute=True)
        assert pmhc2.hla_allele == "HLA-A0101"
        assert pmhc2.allele == "HLA-A*01:01"  # NOTE: Eager impute fails here.  
        
    def test_data_integrity(self):
        # Test to see if the frozen=True holds
        pmhc = pMHC(peptide="GILGFVFTL", hla_allele="HLA-A0201", eager_impute=False)
        # FrozenInstance error should be raised which is a subclass of ValueError
        with pytest.raises(AttributeError):
            pmhc.peptide = "SOME FAKE PEPTIDE"
            pmhc.hla_allele = "HLA-A0101"
            pmhc.allele = "HLA-A*01:01"
        assert pmhc.peptide == "GILGFVFTL"
        assert pmhc.hla_allele == "HLA-A0201"
        assert pmhc.allele == "HLA-A*02:01"
        
    def test_add_tcr(self):
        # Test adding a TCR
        pmhc = pMHC(peptide="GILGFVFTL", hla_allele="HLA-A0201", eager_impute=False)
        tcr = TCR(cdr3b='cASsIRSsYEqYF', trbv='TRBV19*1', trbj='TRBJ2-07*1',
                  cdr3a="CATGLTGGGNKLTF", trav="TRAV17*1", traj="TRAJ10*1")
        pmhc.add_tcr(tcr)
        assert tcr in pmhc.tcrs

    def test_get_tcrs(self):
        # Test get_pMHCs method
        pmhc = pMHC(peptide="GILGFVFTL", hla_allele="HLA-A0201", eager_impute=False)
        tcr = TCR(cdr3b='CASSIRSSYEQF', trbv='TRBV19*1', trbj='TRBJ2-07*1')
        pmhc.add_tcr(tcr)
        assert pmhc.get_tcrs() == {tcr}

    def test_equality(self):
        # Test equality
        pmhc1 = pMHC(peptide="GILGFVFTL", hla_allele="HLA-A0201", eager_impute=False)
        pmhc2 = pMHC(peptide="GILGFVFTL", hla_allele="HLA-A*02:01", eager_impute=False)
        assert pmhc1 == pmhc2
        assert pmhc1 is not pmhc2
        assert pmhc1 is not None
        
    def test_hash(self):
        # Test hash
        pmhc1 = pMHC(peptide="GILGFVFTL", hla_allele="HLA-A0201", eager_impute=False)
        pmhc2 = pMHC(peptide="GILGFVFTL", hla_allele="HLA-A*02:01", eager_impute=False)
        assert hash(pmhc1) == hash(pmhc2)
        assert hash(pmhc1) is not hash(pmhc2)
        assert hash(pmhc1) is not None

    def test_repr(self):
        # Test repr
        pmhc = pMHC(peptide="GILGFVFTL", hla_allele="HLA-A0201", eager_impute=False)
        assert eval(repr(pmhc)) == pmhc

    def test_str(self):
        # Test str
        pmhc = pMHC(peptide="GILGFVFTL", hla_allele="HLA-A0201", eager_impute=False)
        assert str(pmhc) == "GILGFVFTL[SEP]YFAMYGEKVAHTHVDTLYVRYHYYTWAVLAYTWY"

    def test_check_mutations(self):
        # Test static method

        og_seq = "MAVMAPRTLVLLLSGALALTQ"
        real_mutations = [(5, "P", "Q"), (2, "V", "W"), (20, "Q", "P")]
        fake_mutations = [(1, "A", "R"), (20, "L", "S")]
        null_mutations = []
        assert pMHC.check_mutations(real_mutations, og_seq)
        assert not pMHC.check_mutations(fake_mutations, og_seq)
        assert pMHC.check_mutations(null_mutations, og_seq)

    def test_apply_mutations(self):
        og_seq = "MAVMAPRTLVLLLSGALALTQ"
        real_mutations = [(5, "P", "Q"), (2, "V", "W"), (20, "Q", "P")]
        null_mutations = []
        assert pMHC.apply_mutations(null_mutations, og_seq) == og_seq
        assert (
            pMHC.apply_mutations(real_mutations, og_seq) == "MAWMAQRTLVLLLSGALALTP"
        )

    def test_allele2seq(self):
        a1 = pMHC("GILGFVFTL", "HLA-A*02:01")
        a2 = pMHC("GILGFVFTL", "HLA-A*02:01 K66A E63Q mutant")
        a3 = pMHC("GILGFVFTL", "HLA-B*35:08")
        a4 = pMHC("GILGFVFTL", "HLA-B*35:08 Q65A T69A Q155A mutant")
        a5 = pMHC("GILGFVFTL", "HLA-A*02:01 B66A T63Q mutant")

        assert (
            a1.mhc
            == "MAVMAPRTLVLLLSGALALTQTWAGSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDGETRKVKAHSQTHRVDLGTLRGYYNQSEAGSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLENGKETLQRTDAPKTHMTHHAVSDHEATLRCWALSFYPAEITLTWQRDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGQEQRYTCHVQHEGLPKPLTLRWEPSSQPTIPIVGIIAGLVLFGAVITGAVVAAVMWRRKSSDRKGGSYSQAASSDSAQGSDVSLTACKV"
        )
        assert (
            a2.mhc
            == "MAVMAPRTLVLLLSGALALTQTWAGSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDGQTRAVKAHSQTHRVDLGTLRGYYNQSEAGSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLENGKETLQRTDAPKTHMTHHAVSDHEATLRCWALSFYPAEITLTWQRDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGQEQRYTCHVQHEGLPKPLTLRWEPSSQPTIPIVGIIAGLVLFGAVITGAVVAAVMWRRKSSDRKGGSYSQAASSDSAQGSDVSLTACKV"
        )
        assert (
            a3.mhc
            == "MRVTAPRTVLLLLWGAVALTETWAGSHSMRYFYTAMSRPGRGEPRFIAVGYVDDTQFVRFDSDAASPRTEPRAPWIEQEGPEYWDRNTQIFKTNTQTYRESLRNLRGYYNQSEAGSHIIQRMYGCDLGPDGRLLRGHDQSAYDGKDYIALNEDLSSWTAADTAAQITQRKWEAARVAEQRRAYLEGLCVEWLRRYLENGKETLQRADPPKTHVTHHPVSDHEATLRCWALGFYPAEITLTWQRDGEDQTQDTELVETRPAGDRTFQKWAAVVVPSGEEQRYTCHVQHEGLPKPLTLRWEPSSQSTIPIVGIVAGLAVLAVVVIGAVVATVMCRRKSSGGKGGSYSQAASSDSAQGSDVSLTA"
        )
        assert (
            a4.mhc
            == "MRVTAPRTVLLLLWGAVALTETWAGSHSMRYFYTAMSRPGRGEPRFIAVGYVDDTQFVRFDSDAASPRTEPRAPWIEQEGPEYWDRNTAIFKANTQTYRESLRNLRGYYNQSEAGSHIIQRMYGCDLGPDGRLLRGHDQSAYDGKDYIALNEDLSSWTAADTAAQITQRKWEAARVAEARRAYLEGLCVEWLRRYLENGKETLQRADPPKTHVTHHPVSDHEATLRCWALGFYPAEITLTWQRDGEDQTQDTELVETRPAGDRTFQKWAAVVVPSGEEQRYTCHVQHEGLPKPLTLRWEPSSQSTIPIVGIVAGLAVLAVVVIGAVVATVMCRRKSSGGKGGSYSQAASSDSAQGSDVSLTA"
        )
        assert (
            a5.mhc
            == "MAVMAPRTLVLLLSGALALTQTWAGSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDGETRKVKAHSQTHRVDLGTLRGYYNQSEAGSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLENGKETLQRTDAPKTHMTHHAVSDHEATLRCWALSFYPAEITLTWQRDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGQEQRYTCHVQHEGLPKPLTLRWEPSSQPTIPIVGIIAGLVLFGAVITGAVVAAVMWRRKSSDRKGGSYSQAASSDSAQGSDVSLTACKV"
        )

    def test_allele2pseudo(self):
        assert (
            pMHC("GILGFVFTL", "HLA-A*02:174").pseudo == "YFAMYGEKVAHTHVDTLYVRYHYYTWAVLAYTWY"
        )
        assert pMHC("GILGFVFTL", "HLA-A*02:01").pseudo == "YFAMYGEKVAHTHVDTLYVRYHYYTWAVLAYTWY"
        assert (
            pMHC("GILGFVFTL", "HLA-A*02:01 K66A E63Q mutant").pseudo
            == "YFAMYGEKVAHTHVDTLYVRYHYYTWAVLAYTWY"
        )
        assert (
            pMHC("GILGFVFTL", "HLA-A*01:69").pseudo == ""
        )  # Is not present in the full version

    def test_end_to_end_mutations(self):
        # Test apply_mutations
        pmhc = pMHC(peptide="GILGFVFTL", hla_allele="HLA-B*08:01 N80I mutant", eager_impute=False)
        assert pmhc.peptide == "GILGFVFTL"
        assert pmhc.hla_allele == "HLA-B*08:01 N80I mutant"
        assert pmhc.allele == "HLA-B*08:01 N80I mutant"
        assert pmhc.pseudo == "YDSEYRNIFTNTDESNLYLSYNYYTWAVDAYTWY"
        assert pmhc.mhc == ("MLVMAPRTVLLLLSAALALTETWAGSHSMRYFDTAMSRPGRGEPRFISVGYVDDTQFVRFDSDAA"
                            "SPREEPRAPWIEQEGPEYWDRNTQIFKTNTQTDRESLRILRGYYNQSEAGSHTLQSMYGCDVGPD"
                            "GRLLRGHNQYAYDGKDYIALNEDLRSWTAADTAAQITQRKWEAARVAEQDRAYLEGTCVEWLRRY"
                            "LENGKDTLERADPPKTHVTHHPISDHEATLRCWALGFYPAEITLTWQRDGEDQTQDTELVETRPA"
                            "GDRTFQKWAAVVVPSGEEQRYTCHVQHEGLPKPLTLRWEPSSQSTVPIVGIVAGLAVLAVVVIGA"
                            "VVAAVMCRRKSSGGKGGSYSQAACSDSAQGSDVSLTA")
        
    def test_add_references(self):
        # Test adding a reference
        tcr = TCR(cdr3b='cASsIRSsYEqYF', trbv='TRBV19*1', trbj='TRBJ2-07*1',
                  cdr3a="CATGLTGGGNKLTF", trav="TRAV17*1", traj="TRAJ10*1")
        tcr.add_reference("Ref1")
        assert "Ref1" in tcr.references
        assert len(tcr.references)==1
        tcr.add_reference("Ref2")
        tcr.add_reference("Ref3")
        assert "Ref2" in tcr.references
        assert "Ref3" in tcr.references
        assert len(tcr.references)==3
            
    def test_get_references(self):
        # Test get_references method
        tcr = TCR(cdr3b='cASsIRSsYEqYF', trbv='TRBV19*1', trbj='TRBJ2-07*1',
                  cdr3a="CATGLTGGGNKLTF", trav="TRAV17*1", traj="TRAJ10*1",
                  reference="Ref1")
        assert tcr.get_references() == {"Ref1"}