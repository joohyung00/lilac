"""(Version 1)
Please provide your explanation first, then return an answer by ’Therefore, the answer is:’"""



INSTRUCTION_PROMPT_NONIMAGE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>Using the f_answers() API, you can return a list of answers to a question that can be answered using the retrieved components of webpages.
A retrieved component can be a passage or a table.
Strictly follow the format of the below example. 
Return a SHORT answer to the question using the given evidences, using f_answers() API.
* For yes/no questions, only answer f_answers(["yes"]) of f_answers(["no"]).

"""


INSTRUCTION_PROMPT_IMAGE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>Using the f_answers() API, you can return a list of answers to a question that can be answered using the retrieved components of webpages.
A retrieved component can be a passage, a table, or an image.
Strictly follow the format of the below example.
Return a SHORT answer to the question using the given evidences, using f_answers() API.
* For yes/no questions, only answer f_answers(["yes"]) of f_answers(["no"]).

"""




DEMONSTRATION_PROMPT = """
/*
[Table]
Title: Gonzalo Higuain
Section: Career statistics | Club

Club | Season | League - Division | League - Apps | League - Goals | National Cup - Apps | National Cup - Goals | League Cup - Apps | League Cup - Goals | Continental - Apps | Continental - Goals | Other - Apps | Other - Goals | Total - Apps | Total - Goals 
 River Plate | 2004–05 | Argentine Primera División | 4 | 0 | 0 | 0 | — | — | 0 | 0 | — | — | 4 | 0
River Plate | 2005–06 | Argentine Primera División | 14 | 5 | 0 | 0 | — | — | 4 | 2 | — | — | 18 | 7
River Plate | 2006–07 | Argentine Primera División | 17 | 8 | 0 | 0 | — | — | 2 | 0 | — | — | 19 | 8
River Plate | Total | Total | 35 | 13 | 0 | 0 | — | — | 6 | 2 | 0 | 0 | 41 | 15
Real Madrid | 2006–07 | La Liga | 19 | 2 | 2 | 0 | — | — | 2 | 0 | — | — | 23 | 2
Real Madrid | 2007–08 | La Liga | 25 | 8 | 4 | 1 | — | — | 5 | 0 | — | — | 34 | 9
Real Madrid | 2008–09 | La Liga | 34 | 22 | 2 | 1 | — | — | 7 | 0 | 1 | 1 | 44 | 24
Real Madrid | 2009–10 | La Liga | 32 | 27 | 1 | 0 | — | — | 7 | 2 | — | — | 40 | 29
Real Madrid | 2010–11 | La Liga | 17 | 10 | 2 | 1 | — | — | 6 | 2 | — | — | 25 | 13
Real Madrid | 2011–12 | La Liga | 35 | 22 | 5 | 1 | — | — | 12 | 3 | 2 | 0 | 54 | 26
Real Madrid | 2012–13 | La Liga | 28 | 16 | 5 | 0 | — | — | 9 | 1 | 2 | 1 | 44 | 18
Real Madrid | Total | Total | 190 | 107 | 21 | 4 | — | — | 48 | 8 | 5 | 2 | 264 | 121
Napoli | 2013–14 | Serie A | 32 | 17 | 5 | 2 | — | — | 9 | 5 | — | — | 46 | 24
Napoli | 2014–15 | Serie A | 37 | 18 | 4 | 1 | — | — | 16 | 8 | 1 | 2 | 58 | 29
Napoli | 2015–16 | Serie A | 35 | 36 | 2 | 0 | — | — | 5 | 2 | — | — | 42 | 38
Napoli | Total | Total | 104 | 71 | 11 | 3 | — | — | 30 | 15 | 1 | 2 | 146 | 91
Juventus | 2016–17 | Serie A | 38 | 24 | 4 | 3 | — | — | 12 | 5 | 1 | 0 | 55 | 32
Juventus | 2017–18 | Serie A | 35 | 16 | 4 | 2 | — | — | 10 | 5 | 1 | 0 | 50 | 23
Juventus | 2019–20 | Serie A | 15 | 4 | 0 | 0 | — | — | 5 | 2 | 1 | 0 | 21 | 6
Juventus | Total | Total | 88 | 44 | 8 | 5 | — | — | 27 | 12 | 3 | 0 | 126 | 61
Milan (loan) | 2018–19 | Serie A | 15 | 6 | 1 | 0 | — | — | 5 | 2 | 1 | 0 | 22 | 8
Chelsea (loan) | 2018–19 | Premier League | 14 | 5 | 2 | 0 | 1 | 0 | 2 | 0 | — | — | 19 | 5
Career total | Career total | Career total | 446 | 246 | 43 | 12 | 1 | 0 | 117 | 40 | 10 | 4 | 618 | 301
*/

/*
[Passage]
Title: 2006 FIFA Club World Cup Final

The match pitted Internacional of Brazil, the CONMEBOL club champions, against Barcelona of Spain, the UEFA club champions. Internacional won 1–0, after a counter-attack led by Iarley and the goal scored by Adriano Gabiru at the 82nd minute, in a match watched by 67,128 people. In doing so, Internacional won their first FIFA Club World Cup/Intercontinental Cup and Barcelona remained without any world club title. Deco was named as man of the match.
*/

/*
[Passage]
Title: 2018 UEFA Champions League Final

The 2018 UEFA Champions League Final was the final match of the 2017–18 UEFA Champions League, the 63rd season of Europe's premier club football tournament organised by UEFA, and the 26th season since it was renamed from the European Cup to the UEFA Champions League. It was played at the NSC Olimpiyskiy Stadium in Kiev, Ukraine on 26 May 2018, between Spanish side and defending champions Real Madrid, who had won the competition in each of the last two seasons, and English side Liverpool.
*/

Question = What club that Gonzalo Higuain played for in 2006-07 is in the champions league final?
Explanation = We first have to locate what clubs Gonzalo Higuain played in in 2006-07. We can check that there is a table of Gonzalo Higuain's career statistics. In 2006-07, he played for River Plate and Real Madrid. We then have to check which of these clubs played in the Champions League final. We can find that Real Madrid played in the 2018 UEFA Champions League Final. Therefore, the answer is: f_answers(["Real Madrid"])
                                                                                                                                                                                                                                 


/*
[Passage]
Title: South Asia

The current territories of Afghanistan, Bangladesh, Bhutan, Maldives, Nepal, India, Pakistan, and Sri Lanka form South Asia. The South Asian Association for Regional Cooperation (SAARC) is an economic cooperation organisation in the region which was established in 1985 and includes all eight nations comprising South Asia.
*/

/*
[Passage]
Title: UK Joint Expeditionary Force

The UK Joint Expeditionary Force (JEF) is a United Kingdom-led expeditionary force which may consist of, as necessary, Denmark, Finland, Estonia, Latvia, Lithuania, the Netherlands, Sweden and Norway. It is distinct from the similarly named Franco-British Combined Joint Expeditionary Force.
*/

/*
[Table]
Title: Lithuanian Armed Forces
Section: Current operations

Deployment | Organization | Operation | Personnel 
 Somalia | EU | Operation Atalanta | 15
Mali | EU | EUTM Mali | 2
Afghanistan | NATO | Operation Resolute Support | 29
Libya | EU | EU Navfor Med | 3
Mali | UN | MINUSMA | 39
Iraq | CJTF | Operation Inherent Resolve | 6
Central African Republic | EU | EUFOR RCA | 1
Kosovo | NATO | KFOR | 1
Ukraine |  | Training mission | 40
*/

Question = Among the Lithuanian Armed Forces' current operations, which of her deployments involves fewer personnel: Kosovo, or the deployment in the nation that, along with six others, constitutes the subcontinent of South Asia?
Explanation = The South Asia passage identifies Afghanistan as part of the subcontinent. The Lithuanian Armed Forces' operations table shows 29 personnel deployed in Afghanistan. In contrast, only 1 personnel is stationed in Kosovo. 1 is fewer than 29. Therefore, the answer is: f_answers(["Kosovo"]).



"""





PAGE_PROMPT = """

Using the images and the texts given, answer the question in a single word or phrase. 

Question: {question}
Answer:"""