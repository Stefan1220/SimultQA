'''
define templates to retrieve path based on the topic entity
'''
import json
import re
import time
from collections import defaultdict
import sys
sys.path.append('../freebase/')
from freebase import Freebase
freebase_url = "http://164.107.116.56:8890/sparql"
freebase = Freebase(freebase_url)

const2rel = {'largest': """ns:location.location.area ns:topic_server.population_number""",
             'biggest': """ns:location.location.area ns:topic_server.population_number""",
             'most': """ns:location.location.area ns:topic_server.population_number ns:military.casualties.lower_estimate
               ns:location.religion_percentage.percentage ns:geography.river.discharge ns:aviation.airport.number_of_runways""",
             'major': """ns:location.location.area ns:topic_server.population_number ns:military.casualties.lower_estimate
               ns:location.religion_percentage.percentage ns:geography.river.discharge ns:aviation.airport.number_of_runways""",
             'predominant': """ns:location.religion_percentage.percentage""",
             'warmest': """ns:travel.travel_destination_monthly_climate.average_max_temp_c""",
             'tallest': """ns:architecture.structure.height_meters"""}

m2n_mappings = json.load(open('/data/mo.169/CQD4QA/candidates_data/kb/candidate_path/data/m2n_cache.json'))

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False

def is_valid_date(strdate):
    try:
        if ":" in strdate:
            time.strptime(strdate, "%Y-%m-%d %H:%M:%S")
        else:
            time.strptime(strdate, "%Y-%m-%d")
        return True
    except:
        return False

def visit_freebase(query, te_num, hop_num):

	temp_result = freebase.sparql_query(query)
	results = temp_result['results']['bindings'] # [{'r2':{'value':}, ..}}, ]
	parse_results_tmp = {}
	for result in results:

		path_list = []
		entity_mid = '' # incorporate entity info
		et_position_mark = False
		# e_position_mark = False
		for index, item in enumerate(result):
			if index == 0 and item == 'et':
				et_position_mark = True			
			if hop_num == 'hop1':

				if item == 'e':
					mid = result[item]['value'].split('/')[-1]
				if item == 'et':
					entity_mid = result[item]['value'].split('/')[-1]
			
			if hop_num == 'hop2':
				if te_num == 'single':
					if item == 'e':
						mid = result[item]['value'].split('/')[-1]
					if item == 'et':
						entity_mid = ''

				elif te_num == 'multi':
					if item == 'e':
						entity_mid = result[item]['value'].split('/')[-1]
						# entity_mid = ''
						if index == 0:
							et_position_mark = True
						else:
							et_position_mark = False
					if item == 'et':
						mid = result[item]['value'].split('/')[-1]
			
			if item.startswith('r'):
				relation = result[item]['value'].split('/')[-1].split('.')[-1]
				if '_' in relation:
					path_list.extend(relation.split('_'))
				else:
					path_list.append(relation)

		try:
			entity_name = m2n_mappings[entity_mid]
		except Exception as e:
			# print('Error: ', e)
			entity_name = entity_mid
		if et_position_mark:
			path = entity_name  + ' ' + ' '.join(path_list) # incorporate entity info into path
		else:
			path = ' '.join(path_list) + ' ' + entity_name
		# path = (entity_mid, ' '.join(path_list) )
		if path_list == [] or mid == '':
			continue
		else:
			if path not in parse_results_tmp:
				parse_results_tmp[path] = [mid]
			else:
				if mid not in parse_results_tmp[path]:
					parse_results_tmp[path].append(mid)
	
	parse_results = parse_results_tmp

	return parse_results

def SPARQL_1hop(te, nte, superlative): # te => topic entity list; nte => entity but not topic entity
	retu = "?et, ?r, ?e"
	#const = "FILTER (?e!=%s)\nFILTER (!isLiteral(?e) OR lang(?e) = '' OR langMatches(lang(?e), 'en'))"%(te)
	const = "FILTER (!isLiteral(?e) OR lang(?e) = '' OR langMatches(lang(?e), 'en'))"
	const1 = "?e ns:type.object.name ?name."
	value1 = "VALUES ?et {%s}"%(te)

	sparql_txt = """PREFIX ns:<http://rdf.basekb.com/ns/>\nSELECT DISTINCT %s WHERE {%s\n%s\n%s""" %(retu, const, const1, value1)
	if nte != 'ns:':
		value2 = "\nVALUES ?e {%s}"%(nte)
		sparql_txt += value2

	if superlative in ['largest', 'most', 'predominant', 'biggest', 'major', 'warmest', 'tallest']:
		sparql_txt += "\nVALUES ?r {%s}" %const2rel[superlative]
		trips = "\n?et ?r ?e}"
		sparql_txt += trips
		sparql_txt += "\nORDER BY DESC(xsd:integer(?e))\nLIMIT 1"
	else:
		trips = "\n?et ?r ?e}"
		sparql_txt += trips
	return sparql_txt

def SPARQL_1hop_type(te, type_e):
	retu = "?et, ?r, ?rtype, ?etype, ?e"
	const = "FILTER (!isLiteral(?e) OR lang(?e) = '' OR langMatches(lang(?e), 'en'))"
	const1 = "?e ns:type.object.name ?name."
	value1 = "VALUES ?et {%s}"%(te)
	value2 = "VALUES ?rtype {%s}"%("ns:common.topic.notable_types")
	trips = "?et ?r ?e . ?e ?rtype ?etype"

	value3 = "VALUES ?etype {%s}"%(type_e)
	sparql_txt = """PREFIX ns:<http://rdf.basekb.com/ns/>\nSELECT DISTINCT %s WHERE {%s\n%s\n%s\n%s\n%s\n%s}""" %(retu, const, const1, value1, value2, value3, trips)

	return sparql_txt

def SPARQL_2hop(te, nte, year, superlative): # contain CVT node
	retu = "?et, ?r, ?r1, ?e"
	const = "FILTER (!isLiteral(?e) OR lang(?e) = '' OR langMatches(lang(?e), 'en'))"
	const1 = "MINUS {?d ns:type.object.name ?name.}."
	value1 = "VALUES ?et {%s}"%(te)

	sparql_txt = """PREFIX ns:<http://rdf.basekb.com/ns/>\nSELECT DISTINCT %s WHERE {%s\n%s\n%s""" %(retu, const, const1, value1)
	if nte != 'ns:':
		sparql_txt += "\nVALUES ?e {%s}"%(nte)
	if year != '':
		sparql_txt += "\nFILTER(?d >= xsd:date('%s-01-01') && ?d < xsd:date('%s-01-01'))"%(year, year)
	if superlative in ['largest', 'most', 'predominant', 'biggest', 'major', 'warmest', 'tallest']:
		sparql_txt += "\nVALUES ?r {%s}" %const2rel[superlative]
		trips = "\n?et ?r ?d. ?d ?r1 ?e}"
		sparql_txt += trips
		sparql_txt += "\nORDER BY DESC(xsd:integer(?e))\nLIMIT 1"
	else:
		trips = "\n?et ?r ?d. ?d ?r1 ?e}"
		sparql_txt += trips

	return sparql_txt

def SPARQL_2hop_type(te, type_e):
	retu = "?et, ?r, ?r1, ?rtype, ?etype, ?e"
	const = "FILTER (!isLiteral(?e) OR lang(?e) = '' OR langMatches(lang(?e), 'en'))"
	const1 = "?e ns:type.object.name ?name."
	const2 = "MINUS {?d ns:type.object.name ?name.}."
	value1 = "VALUES ?et {%s}"%(te)
	value2 = "VALUES ?rtype {%s}"%("ns:common.topic.notable_types")

	trips = "?et ?r ?d . ?d ?r1 ?e . ?e ?rtype ?etype"

	value3 = "VALUES ?etype {%s}"%(type_e)
	sparql_txt = """PREFIX ns:<http://rdf.basekb.com/ns/>\nSELECT DISTINCT %s WHERE {%s\n%s\n%s\n%s\n%s\n%s\n%s}""" %(retu, const, const1, const2, value1, value2, value3, trips)

	return sparql_txt

def SPARQL_2hop_cons(te, cons_e, nte, year): # CVT node along with a constraint; cons_e => entity in constraint
	retu = "?et, ?r, ?r1, ?r2, ?e1, ?e"
	const = "FILTER (!isLiteral(?e) OR lang(?e) = '' OR langMatches(lang(?e), 'en'))"
	const1 = "MINUS {?d ns:type.object.name ?name.}."
	value1 = "VALUES ?et {%s}"%(te)
	value2 = "VALUES ?e1 {%s}"%(cons_e)
	
	sparql_txt = """PREFIX ns:<http://rdf.basekb.com/ns/>\nSELECT DISTINCT %s WHERE {%s\n%s\n%s\n%s""" %(retu, const, const1, value1, value2)
	if nte != 'ns:':
		sparql_txt += "\nVALUES ?e {%s}"%(nte)
	if year != '':
		sparql_txt += "\nFILTER(?d >= xsd:date('%s-01-01') && ?d < xsd:date('%s-01-01'))"%(year, year)

	trips = "\n?et ?r ?d . ?d ?r1 ?e . ?d ?r2 ?e1}"
	sparql_txt += trips
	return sparql_txt

def SPARQL_1hop_reverse(te, nte):
	retu = "?r, ?et, ?e"
	const = "FILTER (!isLiteral(?e) OR lang(?e) = '' OR langMatches(lang(?e), 'en'))"
	const1 = "?e ns:type.object.name ?name."
	trips = "?e ?r ?et"
	value1 = "VALUES ?et {%s}"%(te)
	if nte != 'ns:':
		value2 = "VALUES ?e {%s}"%(nte)
		sparql_txt = """PREFIX ns:<http://rdf.basekb.com/ns/>\nSELECT DISTINCT %s WHERE {%s\n%s\n%s\n%s\n%s}""" %(retu, const, const1, value1, value2, trips)

	else:
		sparql_txt = """PREFIX ns:<http://rdf.basekb.com/ns/>\nSELECT DISTINCT %s WHERE {%s\n%s\n%s\n%s}""" %(retu, const, const1, value1, trips)
	return sparql_txt

def SPARQL_2hop_reverse(te, nte, year):
	retu = "?r, ?r1, ?et, ?e"
	const = "FILTER (!isLiteral(?e) OR lang(?e) = '' OR langMatches(lang(?e), 'en'))"
	const1 = "MINUS {?d ns:type.object.name ?name.}."
	value1 = "VALUES ?et {%s}"%(te)

	sparql_txt = """PREFIX ns:<http://rdf.basekb.com/ns/>\nSELECT DISTINCT %s WHERE {%s\n%s\n%s""" %(retu, const, const1, value1)
	if nte != 'ns:':
		sparql_txt += "\nVALUES ?e {%s}"%(nte)
	if year != '':
		sparql_txt += "\nFILTER(?d >= xsd:date('%s-01-01') && ?d < xsd:date('%s-01-01'))"%(year, year)

	trips = "\n?e ?r ?d. ?d ?r1 ?et}"
	sparql_txt += trips
	
	return sparql_txt

def retrieve_cand_path(hop_num, te_list, nte_list, cons_e_list, type_e, year, superlative): # te => topic entity, nte => the second entity like in the intersection questions, cons_e => constraint entity

	te_num = 'single'
	if len(te_list) > 1:
		te_num = 'multi'
	
	superlative = ''
	year = ''
	if hop_num == 'hop1':
		year = ''
		superlative = ''

	te = ''
	if len(te_list) > 0:
		if len(te_list) > 100:
			te_list = te_list[0:100]
		for item in te_list:
			te += 'ns:' + item + ' ' 
	else:
		te = 'ns:'+ te

	# constraint entity
	cons_e = ''
	if len(cons_e_list) > 0:
		if len(cons_e_list) > 100:
			cons_e_list = cons_e_list[0:100]
		for item in cons_e_list:
			cons_e += 'ns:' + item + ' ' 
	else:
		cons_e = 'ns:'+ cons_e

	nte = ''
	if hop_num == 'hop2' and te_num == 'multi':
		nte = cons_e
	else:
		nte = 'ns:' + nte

	type_e = 'ns:' + type_e

	cand_path = []
	query1 = SPARQL_1hop(te, nte, superlative)
	query2 = SPARQL_2hop(te, nte, year, superlative)
	query3 = SPARQL_1hop_reverse(te, nte)
	query4 = SPARQL_2hop_reverse(te, nte, year)
	if type_e != 'ns:':
		query5 = SPARQL_1hop_type(te, type_e)
		query6 = SPARQL_2hop_type(te, type_e)
	if cons_e != 'ns:':
		query7 = SPARQL_2hop_cons(te, cons_e, nte, year)

	for i in range(1, 8):

		if cons_e == 'ns:' and i == 7:
			continue
		if type_e == 'ns:' and (i == 5 or i == 6):
			continue

		result = visit_freebase(eval('query%s'%(i)), te_num, hop_num) # result: dict
		cand_path.append(result)

	cand_path_final_dict = {}
	cand_path_final = []
	for cand in cand_path:
		for key in cand:
			if key not in cand_path_final_dict:
				cand_path_final_dict[key] = cand[key]
				cand_path_final.append([key, list(set(cand[key]))])
			else:
				if cand_path_final_dict[key] != cand[key]:
					cand_path_final.append([key, list(set(cand[key]))])

	return cand_path_final

def main():
	dataset_str = 'dev'

	with open('/data/mo.169/CQD4QA/85/HybridQA/data/cwq_%s_class_entity_no_assume_superlative.txt'%(dataset_str), 'r') as f:
		class_entities = f.readlines()

	with open('/data/mo.169/CQD4QA/85/HybridQA/data/cwq_%s_gold_paths.json'%(dataset_str), 'r') as f:
		gold_paths = json.load(f)

	save_path = '/data/mo.169/CQD4QA/85/HybridQA/data/cross_data/cwq_%s_cand_paths_no_assume_with_entity_superlative.txt'%(dataset_str)
	with open(save_path, 'w') as f:
		for index, class_entity in enumerate(class_entities):
			print(index)
			class_entity = eval(class_entity)

			cand_path = []

			# first subq
			temp_path_1 = retrieve_cand_path('hop1', class_entity['entity_1'], [], class_entity['entity_cons'], class_entity['entity_type'], class_entity['year'], class_entity['superlative_word']) 
			cand_path.append(temp_path_1)
			# second subq
			# temp_path_2 = retrieve_cand_path(freebase, class_entity['intermediate_ans'], class_entity['entity_2'], class_entity['entity_cons'], class_entity['entity_type']) 
			# cand_path.append(temp_path_2)
			temp_path_2 = []
			cand_path.append(temp_path_2)
			
			output = {}
			output['gold_hop1'] = [gold_paths[str(index)][0], class_entity['intermediate_ans'] ]
			try:
				output['gold_hop2'] = [gold_paths[str(index)][1], [] ]
			except:
				output['gold_hop2'] = ['', [] ]	

			output['cand_hop1'] = cand_path[0]
			output['cand_hop2'] = cand_path[1]

			f.write(str(output)+'\n')

if __name__ == '__main__':
	main()
