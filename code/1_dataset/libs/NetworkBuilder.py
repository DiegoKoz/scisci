import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np


class NetworkBuilder:
    def __init__(self, df, references):
        references = references.drop_duplicates(['eid','eid_of_ref'])
        metadata = df.copy()
        metadata = metadata.drop_duplicates(['eid'])
        metadata.loc[:,'coverDate'] = metadata.loc[:,'coverDate'].apply(lambda x: int(x.rsplit('-')[0]))
        max_year = max(metadata['coverDate'].values)
        metadata.reset_index(drop=True, inplace=True)
        self.max_year = max_year
        self.metadata = metadata
        self.references = references
        
    def extract_id(self,l,pos=0):
        if l is None:
            return [np.nan]
        else:
            return list(map(lambda x: x[pos],l))

    def convert_authkeywords(self,l):
        if l is None:
            return [np.nan]
        else:
            return [l]    
        
    def citations_at_n(self,edge, n):
    
        if type(n) is int:
            _ = edge[edge.time_dif <=n].groupby('cited').count().iloc[:,1].to_dict()

        if n=='all':
            _ = edge.groupby('cited').count().iloc[:,1].to_dict()

        tmp_df = self.metadata.loc[:,['eid','coverDate']]
        tmp_df['max_year'] = self.max_year
        tmp_df['max_span'] = tmp_df['max_year'] -  tmp_df['coverDate']

        if type(n) is int:
            res = np.where(tmp_df['max_span']>=n, [_.get(x) if (x in _.keys()) else 0 for x in tmp_df.eid.values], None)

        if n=='all':
            res = [_.get(x) if (x in _.keys()) else 0 for x in tmp_df.eid.values]
        return res

    def cumulative_ref(self):
        
        metadata_df = self.metadata.loc[:,['coverDate','eid']]

        corpus_eid = self.metadata.eid.unique()
        ref_df = self.filter_references(corpus_eid, self.references)        
        edges = ref_df.loc[:,['eid', 'eid_of_ref']]
        edges.columns = ['citing','cited']

        edges = pd.merge(edges,metadata_df, how='left', left_on=['citing'], right_on=['eid'])
        edges = pd.merge(edges,metadata_df, how='left', left_on=['cited'], right_on=['eid'])
        edges= edges[['citing','cited','coverDate_x','coverDate_y']]
        edge= edges.rename(columns = {"coverDate_x":'citing_year',"coverDate_y":"cited_year"})

        edge['time_dif'] = edge['citing_year'] - edge['cited_year']

        citations_at_list = list(map(lambda x: self.citations_at_n(edge,x), range(10)))

        for i in range(10):
            metadata_df['at_{}'.format(i)] = citations_at_list[i]

        metadata_df['all_citations_internal'] = self.citations_at_n(edge,'all')

        multipliers = {}
        for index in range(3,12):
            multiplier = metadata_df.iloc[: , index].mean()/metadata_df.iloc[: , index-1].mean()
            multipliers[metadata_df.iloc[: ,index].name] = multiplier
            #metadata_df.iloc[: ,index] = metadata_df.apply(lambda x: x.iloc[index] if x.iloc[index] is not None else x.iloc[index -1]*multiplier, axis=1)

        for index in range(3,12):
            metadata_df.iloc[: ,index] = metadata_df.apply(lambda x: x.iloc[index] if x.iloc[index] is not None else x.iloc[index -1]*multipliers[metadata_df.iloc[: ,index].name], axis=1)
        
        metadata_df[metadata_df.columns.drop('eid')] = metadata_df[metadata_df.columns.drop('eid')].apply(pd.to_numeric, errors='coerce')
        
        return metadata_df.iloc[:,1:13]

    def prepare_metadata(self,text_encoding = 'tfidf'):
        """
            text_encoding: str, options {'tfidf', 'd2v', 'bert'}, default: tfidf, 
                           return the text encoded as a TF-IDF document vector.
                           If 'd2v' it computes the Doc2Vec Embedding of the document 
        """
        metadata_df = self.metadata.loc[:,['eid','affiliation','authors','subject_areas','coverDate','topic_dist','tfidf_vec','d2v_vec', 'bert_embedding']]
        metadata_df.loc[:,'affiliation'] = metadata_df.loc[:,'affiliation'].apply(lambda l: self.extract_id(l)[0])
        metadata_df.loc[:,'first_author'] = metadata_df.loc[:,'authors'].apply(lambda l: self.extract_id(l)[0])
        del metadata_df['authors'] 
        metadata_df.loc[:,'subject_areas'] = metadata_df.loc[:,'subject_areas'].apply(lambda l: self.extract_id(l, pos=2))
        metadata_df[['subject_area_1','subject_area_2','subject_area_3']] = pd.DataFrame(metadata_df.subject_areas.tolist()).loc[:,0:2]
        del metadata_df['subject_areas']
               
        topics_df = pd.DataFrame(metadata_df['topic_dist'].values.tolist()).add_prefix('topic_dist')
        metadata_df = pd.concat([metadata_df,topics_df], axis=1)
        del metadata_df['topic_dist']
        
        if text_encoding == 'tfidf':
            tfidf_df = pd.DataFrame(metadata_df['tfidf_vec'].values.tolist()).add_prefix('tfidf')
            metadata_df = pd.concat([metadata_df,tfidf_df], axis=1)
        if text_encoding == 'd2v':
            d2v_df = pd.DataFrame(metadata_df['d2v_vec'].values.tolist()).add_prefix('d2v')
            metadata_df = pd.concat([metadata_df,d2v_df], axis=1)
        if text_encoding == 'bert':
            bert_df = pd.DataFrame(metadata_df['bert_embedding'].values.tolist()).add_prefix('bert')
            metadata_df = pd.concat([metadata_df,bert_df], axis=1)

        del metadata_df['tfidf_vec']
        del metadata_df['d2v_vec']
        del metadata_df['bert_embedding']
            
        #citations at
        _ = self.cumulative_ref()
        metadata_df = pd.merge(metadata_df,_, how='left', left_on=['eid'], right_on=['eid'])
        
        cat_vars = ['affiliation','first_author','subject_area_1','subject_area_2','subject_area_3']
        LE = {}
        for var in cat_vars:
            metadata_df.loc[:,var] = pd.to_numeric(metadata_df.loc[:,var])
            if var in ['affiliation','first_author']:
                
                tmp_le = LabelEncoder()
                metadata_df.loc[:,var] = tmp_le.fit_transform(metadata_df.loc[:,var])
                LE[var] = tmp_le  
        #for the subject area, I use dummies instead of label encoding.
        metadata_df = pd.get_dummies(metadata_df,columns=['subject_area_1','subject_area_2','subject_area_3'])

        X = torch.tensor(metadata_df[metadata_df.columns[metadata_df.columns!='eid']].values)
        X = X.to(torch.float)
        self.metadata_df = metadata_df
        return X,LE #eid
    
    def filter_references(self, corpus_eid, references):
        #remove references without ID
        ref_df = references.copy()
        ref_df = ref_df[ref_df.eid_of_ref.notnull()].reset_index(drop=True)
        # add the '2-s2.0-' of the eid_of_ref
        ref_df['eid_of_ref'] = ref_df['eid_of_ref'].apply(lambda x: '2-s2.0-' + str(int(x)))
        #filter for within corpus eids
        ref_df = ref_df[ref_df['eid_of_ref'].isin(corpus_eid)]
        ref_df = ref_df[ref_df['eid'].isin(corpus_eid)]
        return ref_df
    # p(a->b)
    
    def build_edges(self):
        corpus_eid = self.metadata.eid.unique()
        ref_df = self.filter_references(corpus_eid, self.references)        
        edges = ref_df.loc[:,['eid', 'eid_of_ref']]
        edges.columns = ['citing','cited']
        
        # replace EID to row_index in the metadata_df
        eid_row_indices = self.metadata_df[['eid']]
        eid_row_indices.reset_index(level=0, inplace=True)
        
        edges['citing'] =edges['citing'].map(eid_row_indices.set_index('eid')['index'])
        edges['cited'] =edges['cited'].map(eid_row_indices.set_index('eid')['index'])
        edges['citing'] = edges['citing'].astype(int)
        edges['cited'] =  edges['cited'].astype(int)
        edge_index = torch.tensor(edges.values).t().contiguous()
        return edge_index,eid_row_indices