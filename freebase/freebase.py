import traceback
from SPARQLWrapper import SPARQLWrapper, JSON
import pdb

class Freebase:
  def __init__(self, url):
    self.url = url
    self.sparql = SPARQLWrapper(self.url)
  def sparql_query(self, query):
    self.sparql.setQuery(query)
    self.sparql.setReturnFormat(JSON)
    return self.sparql.query().convert()
  def find_name(self, mid):
    # query = u"PREFIX ns: <http://rdf.basekb.com/ns/>\nSELECT DISTINCT ?x\nWHERE {{\nFILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))\nns:{0} ns:type.object.name ?x\n}}\n".format(mid)
    query = u"PREFIX ns: <http://rdf.basekb.com/ns/>\nSELECT DISTINCT ?x\nWHERE {{\nFILTER (isLiteral(?x) AND lang(?x) = 'en')\nns:{0} ns:type.object.name ?x\n}}\n".format(mid)
    answer = self.sparql_query(query)
    answer_value = [_[u"x"][u"value"] for _ in answer[u"results"][u"bindings"]]
    if len(answer_value) > 0:
      return answer_value
    query = u"PREFIX ns: <http://rdf.basekb.com/ns/>\nSELECT DISTINCT ?x\nWHERE {{\nFILTER (isLiteral(?x) AND lang(?x) = 'en')\nns:{0} ns:common.notable_for.display_name ?x\n}}\n".format(mid)
    answer = self.sparql_query(query)
    answer_value = [_[u"x"][u"value"] for _ in answer[u"results"][u"bindings"]]
    return answer_value
  def find_relation_name(self, relation):
    query = u"PREFIX ns: <http://rdf.basekb.com/ns/>\nSELECT DISTINCT ?x\nWHERE {{\nFILTER (isLiteral(?x) AND lang(?x) = 'en')\nns:{0} ns:type.object.name ?x\n}}\n".format(relation)    
    answer = self.sparql_query(query)
    answer_value = [_[u"x"][u"value"] for _ in answer[u"results"][u"bindings"]]
    return answer_value
  def find_ent_name(self, ent_type):
    query = u"PREFIX ns: <http://rdf.basekb.com/ns/>\nSELECT DISTINCT ?x\nWHERE {{\nFILTER (isLiteral(?x) AND lang(?x) = 'en')\nns:{0} ns:type.object.name ?x\n}}\n".format(ent_type)
    answer = self.sparql_query(query)
    answer_value = [_[u"x"][u"value"] for _ in answer[u"results"][u"bindings"]]
    return answer_value
  def is_cvt(self, mid):
    """
    Check whether a MID is a CVT node.
    :param mid:
    :return:
    """
    # If found name, then is not CVT
    name = self.find_name(mid)
    if name:
      return False
    # If found alias, then is not CVT
    query = u"PREFIX ns: <http://rdf.basekb.com/ns/>\nSELECT DISTINCT ?x\nWHERE {{\nFILTER (isLiteral(?x) AND lang(?x) = 'en')\nns:{0} ns:common.topic.alias ?x\n}}\n".format(mid)
    answer = self.sparql_query(query)
    answer_value = [_[u"x"][u"value"] for _ in answer[u"results"][u"bindings"]]
    if len(answer_value) > 0:
      return False
    # If none of the above if found, then assume to be CVT
    return True


# freebase_url = "http://164.107.116.56:8890/sparql"
# freebase = Freebase(freebase_url)
# query = 'PREFIX ns:<http://rdf.basekb.com/ns/>\nSELECT ?r, ?e1 WHERE {\nVALUES ?r {ns:base.schemastaging.context_name.nickname ns:base.schemastaging.context_name.official_name ns:type.object.name ns:base.aareas.schema.administrative_area.short_name}\nns:m.09nqly6 ?r ?e1}'
# print(freebase.sparql_query(query))

#mid = "m.010vz"
#print(freebase.find_name(mid))




