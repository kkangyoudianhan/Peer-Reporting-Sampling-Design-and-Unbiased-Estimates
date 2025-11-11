GitHub Social Network

Description

A large social network of GitHub developers which was collected from the public API in June 2019. Nodes are developers who have starred at least 10 repositories and edges are mutual follower relationships between them. The vertex features are extracted based on the location, repositories starred, employer and e-mail address. The task related to the graph is binary node classification - one has to predict whether the GitHub user is a web or a machine learning developer. This target feature was derived from the job title of each user.

Properties

- Directed: No.
- Node features: Yes.
- Edge features: No.
- Node labels: Yes. Binary-labeled.
- Temporal: No.
- Nodes: 37,700
- Edges: 289,003
- Density: 0.001 
- Transitvity: 0.013

Possible Tasks

- Binary node classification
- Link prediction
- Community detection
- Network visualization
"F:\zhoumian\ECMR\data\git_web_ml\git_web_ml\musae_git_edges.csv"数据示例：
id_1	id_2
0	23977
1	34526
1	2370
1	14683
1	29982
1	21142
1	20363
1	23830
1	34035
6067	19720
6067	20183
3	4950
3	18029
3	3358
3	34935
3	5916
4	2865
4	9342
5	27803
6	27803
6	18612
6	18876
6	31890
6	17647
6	18562
7	37493
7	33643
7	30199
7	35773
7	11273

"F:\zhoumian\ECMR\data\git_web_ml\git_web_ml\musae_git_features.json"
"F:\zhoumian\ECMR\data\git_web_ml\git_web_ml\musae_git_target.csv"数据示例：
id	name	ml_target
0	Eiryyy	0
1	shawflying	0
2	JpMCarrilho	1
3	SuhwanCha	0
4	sunilangadi2	1
5	j6montoya	0
6	sfate	0
7	amituuush	0
8	mauroherlein	0
9	ParadoxZero	0
10	llazzaro	0
11	beeva-manueldepaz	0
12	damianmuti	0
13	apobbati	0
14	hwlv	0
15	haroldoramirez	0
16	jasonblanchard	0
17	BahiHussein	0
18	itsmevanessi	0
19	nwjsmith	0
20	chengzhipeng	0
21	tiensonqin	0
22	pdokas	0
23	maxfierke	0
24	davidthewatson	0
25	kidbai	0
26	Micanss	0
27	hepin1989	0
