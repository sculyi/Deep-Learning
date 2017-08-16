'''
author lyi
date 20170814
'''
import re
from string import digits
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def parse_cmu(cmufh):
    """Parses an incoming file handle as a CMU pronouncing dictionary file.
       Returns a list of 2-tuples pairing a word with its phones (as a string)"""
    pronunciations = list()
    for line in cmufh:
        line = line.strip()
        if line.startswith(';'): continue
        word, phones = line.split("  ")
        word = re.sub(r'\(\d\)$', '', word.lower())
        phones_list = phones.split(" ")
        pronunciations.append((word.lower(), phones))
    return pronunciations 

class CMUdict(object):
    def __init__(self,cmufh='../resource/cmudict-0.7b'):
        self.pronunciations = parse_cmu(open(cmufh))
    
    def syllable_count(phones):
	return sum([phones.count(i) for i in '012'])

    def phones_for_word(self,find):
	"""Searches a list of 2-tuples (as returned from parse_cmu) for the given
		word. Returns a list of phone strings that correspond to that word."""
	matches = list()
	for word, phones in self.pronunciations:
	    if word == find:
		matches.append(phones)
	return matches
    def phones_for_sentence(self,sentence):
        sentence = sentence.strip('\n').split()
        wp=[]
        for word in sentence:
            fwp = self.phones_for_word(word)
            if fwp == []:
                full_word = False
                break
            wp.append( fwp[0].lower().translate(None, digits) )
        return ' '.join(wp)

    def rhyming_part(self,phones):
	"""Returns the "rhyming part" of a string with phones. "Rhyming part" here
		means everything from the vowel in the stressed syllable nearest the end
		of the word up to the end of the word."""
	idx = 0
	phones_list = phones.split()
	for i in reversed(range(0, len(phones_list))):
	    if phones_list[i][-1] in ('1', '2'):
		idx = i
		break
	return ' '.join(phones_list[idx:])

    def search(self,pattern):
	"""Searches a list of 2-tuples (as returned from parse_cmu) for
		pronunciations matching a given regular expression. (Word boundary anchors
		are automatically added before and after the pattern.) Returns a list of
		matching words."""
	matches = list()
	for word, phones in self.pronunciations:
	    if re.search(r"\b" + pattern + r"\b", phones):
		matches.append(word)
	return matches

    def rhymes(self, word):
	"""Searches a list of 2-tuples (as returned from parse_cmu) for words that
		rhyme with the given word. Returns a list of such words."""
	all_rhymes = list()
	all_phones = self.phones_for_word(word)
	for phones_str in all_phones:
	    part = self.rhyming_part(phones_str)
	    rhymes = self.search(part + "$")
	    all_rhymes.extend(rhymes)
	return [r for r in all_rhymes if r != word]

if __name__ == '__main__':
    import sys
    cmuproc=CMUdict('../resource/cmu/cmudict-0.7b')
    print(cmuproc.phones_for_sentence('teacher'))
    '''
    for line in sys.stdin:
        line = line.strip()
        words = line.split()
        rhymes_list = cmuproc.rhymes(words[-1])
        for rhyme in rhymes_list:
            print rhyme
    '''
