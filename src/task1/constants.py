# NOTE: MODIFY TS ONLY WHEN U WANNA CHANGE THE OVERALL PARAMETERS OF THE DATA.

GENERATION_DELTAS = {
    'motherOf': -1, 'fatherOf': -1,
    'daughterOf': 1, 'sonOf': 1,
    
    'grandmotherOf': -2, 'grandfatherOf': -2,
    'granddaughterOf': 2, 'grandsonOf': 2,
    
    'greatGrandmotherOf': -3, 'greatGrandfatherOf': -3,
    'greatGranddaughterOf': 3, 'greatGrandsonOf': 3,
    
    # same gen
    'sisterOf': 0, 'brotherOf': 0,
    'girlCousinOf': 0, 'boyCousinOf': 0,
    'girlSecondCousinOf': 0, 'boySecondCousinOf': 0,
    
    'auntOf': -1, 'uncleOf': -1,
    'nieceOf': 1, 'nephewOf': 1,
    'greatAuntOf': -2, 'greatUncleOf': -2,
    'secondAuntOf': -1, 'secondUncleOf': -1,
    
    # IMP NOTE: a cousinOnceRemovedOf b means a is the YOUNGER one
    # figured this out by tracing fabian26 manually. thank god for fabian26

    'girlFirstCousinOnceRemovedOf': 1, 
    'boyFirstCousinOnceRemovedOf': 1,
}

# gender of HEAD when they use this relation
# weight is how confident we are (3 = explicit like motherOf, 2 = prefix like girlCousin)

GENDER_EVIDENCE = {
    'motherOf': ('F', 3), 'fatherOf': ('M', 3),
    'daughterOf': ('F', 3), 'sonOf': ('M', 3),
    'grandmotherOf': ('F', 3), 'grandfatherOf': ('M', 3),
    'granddaughterOf': ('F', 3), 'grandsonOf': ('M', 3),
    'greatGrandmotherOf': ('F', 3), 'greatGrandfatherOf': ('M', 3),
    'greatGranddaughterOf': ('F', 3), 'greatGrandsonOf': ('M', 3),
    'sisterOf': ('F', 3), 'brotherOf': ('M', 3),
    'auntOf': ('F', 3), 'uncleOf': ('M', 3),
    'nieceOf': ('F', 3), 'nephewOf': ('M', 3),
    'greatAuntOf': ('F', 3), 'greatUncleOf': ('M', 3),
    'secondAuntOf': ('F', 3), 'secondUncleOf': ('M', 3),
    'girlCousinOf': ('F', 2), 'boyCousinOf': ('M', 2),
    'girlSecondCousinOf': ('F', 2), 'boySecondCousinOf': ('M', 2),
    'girlFirstCousinOnceRemovedOf': ('F', 2), 
    'boyFirstCousinOnceRemovedOf': ('M', 2),
}

# which relations should have symmetric counterparts
# sisterOf(a,b) -> b should have sisterOf or brotherOf pointing to a
SYMMETRIC_GROUPS = {
    'sisterOf': ['sisterOf', 'brotherOf'],
    'brotherOf': ['sisterOf', 'brotherOf'],
    'girlCousinOf': ['girlCousinOf', 'boyCousinOf'],
    'boyCousinOf': ['girlCousinOf', 'boyCousinOf'],
    'girlSecondCousinOf': ['girlSecondCousinOf', 'boySecondCousinOf'],
    'boySecondCousinOf': ['girlSecondCousinOf', 'boySecondCousinOf'],
}

# relations that can be derived from chains of other relations
# grandmotherOf = motherOf + parentOf
DERIVABLE_PATTERNS = {
    'grandmotherOf': [('motherOf', 'motherOf'), ('motherOf', 'fatherOf')],
    'grandfatherOf': [('fatherOf', 'motherOf'), ('fatherOf', 'fatherOf')],
    'grandsonOf': [('sonOf', 'sonOf'), ('sonOf', 'daughterOf')],
    'granddaughterOf': [('daughterOf', 'sonOf'), ('daughterOf', 'daughterOf')],
}

PARENT_RELATIONS = {'motherOf', 'fatherOf'}


CHILD_RELATIONS = {'daughterOf', 'sonOf'}

SIBLING_RELATIONS = {'sisterOf', 'brotherOf'}