"""
Automatically perform numerical analysis of historical linguistics data
and create diagrams according to the Historical Glottometry method of
language genealogy (François & Kalyan 2017; http://hg.hypotheses.org/)

Isaac Stead, October 2021


"Given a candidate subgroup S, let us call p the number of innovations
that properly confirm its cohesion (i.e., those which contain the
whole of the subgroup as well as at least one language outside the
subgroup); and q the number of cross-cutting innovations.  Together, p
and q constitute all the relevant evidence needed to assess a
subgroup’s cohesiveness.  We define κ as follows:
      p + 1
κ = —————
    p + q + 1

Given this definition of cohesiveness, we define the subgroupiness
of a subgroup (called ς ‘sigma’).  In principle, subgroupiness
could simply be conceived as the product of epsilon and kappa —
that is, the number of exclusively shared innovations, weighted by
the subgroup’s cohesiveness rating. However, we wish to add a
final refinement to the results, by adding a strictness parameter
n, that would set the “penalty” for overlapping subgroups:
                                n
         n           (  p + 1  )
ς = ε · κ  = ε · (——————————)
                     (p + q + 1)

"""
from itertools import chain
from pandas import read_csv, DataFrame


def contains(s1, s2):
    larger, smaller = sorted([s1, s2], key=len, reverse=True)
    return set(smaller).issubset(set(larger))


def unique(seqs):
    """Return cross-sequentially unique values from a sequence of
    sequences.
    """
    count = {}
    for el in chain.from_iterable(seqs):
        if el in count:
            count[el] += 1
        else:
            count[el] = 1
    return [k for k, v in count.items() if v == 1]


def grouped_unique(seqs):
    """Find inter-sequentially unique values grouped by sequence.
    Horribly inefficient... there must be a better algorithm for this.
    I can see how to do it efficiently with recursion but not
    iteratively.  Think I have some Lisp code somewhere that does it
    """
    count = {}
    for seq in seqs:
        for el in seq:
            if el in count: 
                count[el] += 1
            else:
                count[el] = 1
    seq_uniques = [k for k, v in count.items() if v == 1]
    groups = []
    for seq in seqs:
        group = []
        for el in seq_uniques:
            if el in seq:
                group.append(el)
        groups.append(group)
    return groups


def flatten(seq):
    return chain.from_iterable(seq)


## Public
def analyse(path):
    """Entry point to this module. Perform a historical glottometry analysis on the
    spreadsheet at `path` and return table.
    """
    fm = FeatureMatrix(path)
    return fm.analyse()


class FeatureMatrix:

    def __init__(self, path):
        frame = read_csv(path, sep="\t")

        lang_feats = {l : list() for l in frame.columns[1:]}
        feat_langs = {i : list() for i in frame["Innovation"]}
    
        for row in frame.itertuples():
            innv = getattr(row, "Innovation")
            for lang in lang_feats.keys():
                if getattr(row, lang) == 1:
                    lang_feats[lang].append(innv)
                    feat_langs[innv].append(lang)

        self.lang_feats = lang_feats
        self.feat_langs = feat_langs
        self.feat_matrix = frame


    @property
    def languages(self):
        return set(self.lang_feats.keys())


    @property
    def features(self):
        return set(self.feat_langs.keys())


    def exclusive(self, languages):
        shared = []
        for feat, langs in self.feat_langs.items():
            if sorted(langs) == sorted(languages):
                shared.append(feat)
        return shared

    
    def supporting(self, languages):
        support = []
        for feat, langs in self.feat_langs.items():
            if set(languages).issubset(set(langs)):
                support.append(feat)
        return support


    def conflicting(self, languages):
        all_fs = set(flatten([fs for l, fs in self.lang_feats.items() if l in languages]))
        diff = all_fs.symmetric_difference(self.supporting(languages))
        result = []
        # Check that identified innovations are shared outside this group
        for feat in diff:
            for lang in self.feat_langs[feat]:
                if lang not in languages:
                    result.append(feat)
                    break
        return result


    def cohesiveness(self, languages):
        n_supporting = len(self.supporting(languages))
        n_conflicting = len(self.conflicting(languages))
        return n_supporting / (n_supporting + n_conflicting)


    def subgroupiness(self, languages):
        # TODO: Allow specification of strictness parameter
        cohesiveness = self.cohesiveness(languages)
        exclusively_shared = len(self.exclusive(languages))
        return cohesiveness * exclusively_shared


    def candidate_subgroups(self):
        """Find all potential subgroups, i.e. groups of languages
        which exclusively share at least one innovation."""
        subgroups = []
        for group in self.feat_langs.values():
            if group not in subgroups:
                subgroups.append(group)
        return subgroups


    def analyse(self, rounding=2):
        """Perform the full glottometric analysis and return a summary dict.
        """
        subgroups = self.candidate_subgroups()
        rows = []
        for group in subgroups:
            rows.append({
                "group": ", ".join(group),
                "exclusive": len(self.exclusive(group)),
                "supporting": len(self.supporting(group)),
                "conflicting": len(self.conflicting(group)),
                "cohesion": round(self.cohesiveness(group), rounding),
                "subgroupiness": round(self.subgroupiness(group), rounding),
            })
        return DataFrame(rows)

