library(tidyverse)
library(ggthemes)
library(ggrepel)
library(stringr)
library(glue)
library(RColorBrewer)
library(gridExtra)
library(extrafont) 
library(scales)
#font_import(pattern = "lmroman*") 

loadfonts()
# Fig 1 Figure degree distribution -----------------------------------------------------

degree <- read_csv('results/network_degree.csv')


degree %>% 
  group_by(degree) %>% 
  count() %>% 
  ggplot(., aes(degree, n))+
  geom_point()+
  scale_x_continuous("Degree",
                     breaks = c(1, 3, 10, 30, 100, 300),
                     trans = "log10") +
  scale_y_continuous("Frequency",
                     breaks = c(1, 3, 10, 30, 100, 300, 1000),
                     trans = "log10")+
  #geom_smooth(color='darkgreen')+
  geom_smooth(method = 'lm',formula = 'y~x^3.459859', color ='blue')+
  theme_minimal()+
  annotate('text', x = 5,y=20000,label = "italic(y) == italic(a)*italic(x)^3.5 ", color='blue', parse = TRUE, size=14, family="LM Roman 10")+
  theme(text = element_text(size = 36, family="LM Roman 10"))

ggsave('results/plots/Fig1.png', dpi = 400, width = 10, height = 10)


# Fig 2 Encoder-Decoder -----------------------------------------------

#made on Latex

# Fig 3 Topic Modeling ---------------------------------------------------
df <-  read_csv('results/topics_dist_by_article.csv')

# I need to normalize at article level
df[1:20] = df[1:20]/rowSums(df[1:20])

df_field <- df %>% 
  mutate(field = case_when(field =='History and Philosophy' ~ 'History and Philosophy',
                           field == 'Education, Communication and Anthropology' ~ 'Other Social Sciences',
                           field == 'Library and Information Sciences' ~ 'Library and\nInformation Sciences',
                           TRUE ~field)) %>% 
  group_by(field) %>% 
  summarise_at(vars(topic_1:topic_20), mean)

topics_dist_total <- df %>% 
  summarise_at(vars(topic_1:topic_20), mean)

df_field_normalized <- df_field
df_field_normalized[2:21] <- df_field_normalized[2:21]/as.numeric(topics_dist_total)


df_field_normalized %>% 
  pivot_longer(topic_1:topic_20,names_prefix = 'topic_', names_to = 'Topic') %>% 
  mutate(Topic= as.numeric(Topic)) %>% 
  #filter(Topic %in% as.character(c(1,3,5,7,10,11))) %>% 
  ggplot(aes(Topic,value, fill = factor(Topic), label=Topic))+
  geom_col()+
  geom_text(position = position_dodge(width = 1),size=6, vjust=-0.1, family="LM Roman 10")+
  facet_wrap(.~field, scales = 'free')+
  theme_minimal()+
  theme(legend.position = 'none',
        text = element_text(size = 28, family="LM Roman 10"))+
  labs(x = 'Topic', y ='Relative Importance')

ggsave('results/plots/Fig3.png', dpi = 400, width = 10, height = 10)


# Fig 4 GNN Embedding  TSNE ---------------------------------------------------------
df <- read_csv('results/articles_tsne_proj.csv')

journals= c('scientometrics','journal of informetrics',
            'research policy','science and public policy',
            'research evaluation','public understanding\nof science',
            'synthese','studies in history and\nphilosophy of science')

journals <- stringr::str_to_title(journals)

df <- df %>% 
  mutate(publicationName = case_when(publicationName=='studies in history and philosophy of science' ~'studies in history and\nphilosophy of science',
                                     publicationName == 'public understanding of science' ~'public understanding\nof science',
                                     TRUE ~ publicationName),
         publicationName= stringr::str_to_title(publicationName),
         publicationName = factor(publicationName, levels =  journals))


colors = c("#351ced","#538ced",
           "#d12f06", "#f56c49",
           "#008709","#2ee63a",
           "#da65f7", "#990099")

show_col(colors)
names(colors) <- journals

ggplot(df, aes(xs_gnn, ys_gnn, color=publicationName, size=citedby_count))+
  geom_point(alpha=.75)+
  stat_ellipse(type = "t", level = .95, segments = 100, size=1)+
  theme_void()+
  # scale_color_gdocs()+
  scale_color_manual(values = colors)+
  # scale_color_viridis_d()+
  # scale_alpha(range = c(0.5,1))+
  scale_size_continuous(range = c(0.001,10))+
  theme(text = element_text(size = 36, family="LM Roman 10"),
        legend.position = 'bottom',legend.box="vertical")+
  guides(color=guide_legend(override.aes = list(size=20),nrow = 3,title = 'Journal'), size = guide_legend(title = 'Citations'), alpha= guide_legend(title = 'Citations'))

ggsave('results/plots/Fig4.png', dpi = 300, width = 10, height = 10, scale = 2)


# Fig 5 Semantic Embeddings T-SNE ----------------------------------------

df_semantic <- df %>% 
  select(-c(X1,xs_gnn,ys_gnn)) %>% 
  pivot_longer(cols = xs_d2v:ys_bert, names_to = c('.value','model'), names_pattern = "(.)s_(.*)" ) %>% 
  mutate(model = case_when(model=='bert'~'BERT',
                           model=='d2v' ~'Doc2Vec',
                           model=='tm' ~'LDA'),
         model = factor(model, levels = c('Doc2Vec', 'BERT', 'LDA')))

p1 <- df_semantic %>% 
  filter(model %in% c('Doc2Vec','LDA')) %>% 
  ggplot(.,aes(x,y,color=publicationName, size=citedby_count, alpha=citedby_count))+
  geom_point()+
  stat_ellipse(type = "t", level = .95, segments = 100)+
  theme_void()+
  # scale_color_gdocs()+
  scale_color_manual(values = colors)+
  # scale_color_viridis_d()+
  scale_alpha(range = c(0.5,1))+
  scale_size_continuous(range = c(0.001,10))+
  theme(panel.background = element_rect(colour = 'grey75'),
        text = element_text(size = 36, family="LM Roman 10"),
        legend.position = 'none',legend.box="vertical")+
  facet_wrap(.~model, scales = 'free')+
  guides(color=guide_legend(override.aes = list(size=20),nrow = 3,title = 'Journal'), size = guide_legend(title = 'Citations'), alpha= guide_legend(title = 'Citations'))

# ggsave('results/plots/tsne_d2v_tm_journal.png', dpi = 400, width = 10, height = 5, scale = 2)

p2 <- df_semantic %>% 
  filter(model %in% c('BERT')) %>% 
  ggplot(.,aes(x,y,color=publicationName, size=citedby_count, alpha=citedby_count))+
  geom_point()+
  stat_ellipse(type = "t", level = .95, segments = 100)+
  theme_void()+
  # scale_color_gdocs()+
  scale_color_manual(values = colors)+
  # scale_color_viridis_d()+
  scale_alpha(range = c(0.5,1))+
  scale_size_continuous(range = c(0.001,10))+
  theme(panel.background = element_rect(colour = 'grey75'),
        text = element_text(size = 36, family="LM Roman 10"),
        legend.position = 'bottom',legend.box="vertical")+
  facet_wrap(.~model, scales = 'free')+
  guides(color=guide_legend(override.aes = list(size=20),nrow = 3,title = 'Journal'), size = guide_legend(title = 'Citations'), alpha= guide_legend(title = 'Citations'))



g <- grid.arrange(p1,p2, layout_matrix = rbind(c(1),c(1),c(2),c(2),c(2),c(2)))

ggsave(plot = g,'results/plots/Fig5.png', dpi = 300, width = 10, height = 15, scale = 2)

# Fig 6 Heatmaps. cosine similarity by collaboration status ----------------------

av_dist_collaboration_df_gnn <- read_csv("results/av_dist_collaboration_df_gnn.csv")
av_dist_collaboration_df_bert <- read_csv("results/av_dist_collaboration_df_bert.csv")
av_dist_collaboration_df_tm <- read_csv("results/av_dist_collaboration_df_tm.csv")

heatmap_collab <- function(data,title,text_size=24, legend_position='right' ){

ord <- c("single_author","internal_colab","institutional_colab","international_colab")
# label <- c('Single Author', 'Internal Colab.', 'Institutional Colab.', 'International Colab.')
label <- c('A', 'B', 'C', 'D')

p <- data %>% 
  mutate(collaboration_status_1 = factor(collaboration_status_1, levels =ord,labels = label ),
         collaboration_status_2 = factor(collaboration_status_2, levels =ord, labels = label ),
         publicationName_1 = case_when(publicationName_1 =='research policy' ~'research\npolicy',
                                       TRUE ~publicationName_1),
         publicationName_2 = case_when(publicationName_2 =='research policy' ~'research\npolicy',
                                       TRUE ~publicationName_2),
         publicationName_1 = str_to_title(publicationName_1),
         publicationName_2 = str_to_title(publicationName_2)) %>% 
  ggplot(aes(collaboration_status_1,collaboration_status_2,fill=cos_sim))+
  theme_minimal()+
  facet_grid( publicationName_1~ publicationName_2)+
  labs(x='',y='')+
  geom_tile()+
  scale_fill_viridis_c()+
  coord_fixed()+
  theme(text = element_text(size=text_size, family="LM Roman 10"),
        strip.text = element_text(size=28),
        # axis.text.x = element_text(angle = 90),
        plot.margin=grid::unit(c(0,0,0,0), "cm"))


  if (legend_position=='right') {
    p <- p +guides(fill = guide_colorbar(title = 'Cosine\nSimilarity',
                                         barwidth = 2,
                                         barheight = 15))
  }
  if(legend_position=='left'){
    
    p <- p + theme(legend.position = 'left') +
      guides(fill = guide_colorbar(title = 'Cosine\nSimilarity',
                                          barwidth = 2,
                                          barheight = 15))
  }
  p
} 

heatmap_collab(av_dist_collaboration_df_gnn)
ggsave(glue('results/plots/Fig6a.png'), dpi=400, width = 10, height = 8)

heatmap_collab(av_dist_collaboration_df_tm, text_size = 31,legend_position = 'left')
 ggsave(glue('results/plots/Fig6b.png'), dpi=400, width = 10, height = 8)

heatmap_collab(av_dist_collaboration_df_bert, text_size = 31, legend_position = 'right')
 ggsave(glue('results/plots/Fig6c.png'), dpi=400, width = 10, height = 8)

# I create the unique figure outside

#layout <- rbind(c(NA,1,1,1,1,NA),c(2,2,2,3,3,3))
#
#g <- grid.arrange(plt1,plt2,plt3, layout_matrix = layout)
#ggsave(plot = g,'results/plots/Fig6.png', dpi = 300, width = 10, height = 15, scale = 2)

# Fig 7 Countries average ----------------------------------------------

  
df <- read_csv('results/country_average_sim.csv')


ggplot(df, aes(GNN,BERT, color = continent, label = first_author_country))+
  geom_point(aes(size=n_citations),alpha=0.75)+
  geom_text_repel(show.legend = F, size=7, family="LM Roman 10")+
  theme_minimal()+
  scale_color_gdocs()+
  # scale_color_colorblind()+
  labs(col="Continent",size="Number of citations")+
  scale_y_continuous(limits = c(0.79,.98))+
  scale_size(range = c(3,10))+
  theme(legend.position = 'bottom',
        legend.box="vertical",
        text = element_text(size = 36,family="LM Roman 10"),
    # axis.text = element_text(size = 18),
    #     axis.title = element_text(size=25),
    #     legend.text=element_text(size=18)
    )+
  guides(color=guide_legend(override.aes = list(size=15)), size = guide_legend())
  

ggsave('results/plots/Fig7.png', dpi = 400, width = 10, height = 7, scale = 2)

# Fig 8 avg_embed_journal -------------------------------------------------------
avg_embed_journal <- read_csv("results/avg_embed_journal.csv")


avg_embed_journal <- avg_embed_journal %>%
pivot_longer(cols = cossim_gnn:cossim_bert,names_to = 'type',values_to = 'cossim',names_prefix ='cossim_') %>%
  mutate(publicationName = case_when(publicationName=='studies in history and philosophy of science' ~'studies in history and\nphilosophy of science',
                                     publicationName=="british journal for the history of science"  ~"british journal\nfor the history of science",
                                     publicationName == "science, technology and human values" ~"science, technology\nand human values",
                                     publicationName=="science, technology and society" ~ "science, technology\nand society",
                                     publicationName=="public understanding of science" ~"public understanding\nof science",
                                     publicationName== "science and technology studies" ~"science and\ntechnology studies",
                                     TRUE ~ publicationName),
         publicationName = stringr::str_to_title(publicationName)) %>% 
  group_by(type) %>% 
  arrange(type, desc(cossim)) %>%
  unite("type_publicationName", type, publicationName, sep = "_", remove = FALSE) %>% 
  mutate(type_publicationName=factor(type_publicationName, levels = type_publicationName),
         type= stringr::str_to_upper(type),
         field = case_when(field =='History and Philosophy' ~ 'History and Philosophy',
                           field == 'Education, Communication and Anthropology' ~ 'Other Social Sciences',
                           field == 'Library and Information Sciences' ~ 'Library and\nInformation Sciences',
                           TRUE ~field),
         mean_citations = n_citations/n)

colors = c("#351ced","#d12f06","#008709","#990099")

fields <- c("Library and\nInformation Sciences", "Management", "Other Social Sciences", "History and Philosophy")
# scales::show_col(colors)
names(colors) <- fields

ggplot(avg_embed_journal,aes(cossim,type_publicationName, color=field, shape =field))+
  geom_point(aes(size=mean_citations))+
  scale_y_discrete(breaks = avg_embed_journal$type_publicationName,
                   labels = avg_embed_journal$publicationName)+
  # scale_color_colorblind()+
  scale_color_manual(values = colors)+
  scale_shape_manual(values = 15:18)+
  scale_size_continuous(range = c(1,12))+
  theme_minimal()+
  theme(legend.position = 'bottom',
        legend.box="vertical",
        text = element_text(size = 24, family="LM Roman 10"),
        legend.text = element_text(size=24),
        legend.title = element_text(size=28),
        legend.key.size = unit(3, "lines"),
        axis.text.x = element_blank(),
        axis.ticks.y = element_line())+
  labs(x = '', y ='', color ='Field', size= 'Mean citations', shape = 'Field')+
  guides(color=guide_legend(override.aes = list(size=10),nrow=2), size = guide_legend(), shape = guide_legend())+
  facet_wrap(.~type, scales = 'free')
  
ggsave('results/plots/Fig8.png', dpi = 400, width = 18, height = 12)


# Fig C.1 Graph Neural Networks framework. ---------------------------------------------------------------

#made on latex

# Fig D1 topic Modeling (journal level) -----------------------------------

df <-  read_csv('results/topics_dist_by_article.csv')

# I need to normalize at article level
df[1:20] = df[1:20]/rowSums(df[1:20])

df_journal <- df %>% 
  group_by(publicationName) %>% 
  summarise_at(vars(topic_1:topic_20), mean)


df_journal[2:21] <- df_journal[2:21]/as.numeric(topics_dist_total)

journals_order <- c("Research Policy",
                    "Science And Public Policy",
                    "Scientometrics",
                    "Journal Of Informetrics",
                    "Synthese",
                    "Social Studies Of Science",
                    "Science And Education",
                    "Studies In History And\nPhilosophy Of Science",
                    "Isis",
                    "Science, Technology\nAnd Society",
                    "British Journal\nFor The History Of Science",
                    "Science And\nTechnology Studies",
                    "Public Understanding\nOf Science",
                    "Science, Technology\nAnd Human Values",
                    "Research Evaluation",
                    "Minerva")

xlabs = c(rbind(seq(1,19,2),paste0('\n',seq(2,20,2))))


df_journal %>% 
  pivot_longer(topic_1:topic_20,names_prefix = 'topic_', names_to = 'Topic') %>% 
  mutate(publicationName = case_when(publicationName=='studies in history and philosophy of science' ~'studies in history and\nphilosophy of science',
                                     publicationName=="british journal for the history of science"  ~"british journal\nfor the history of science",
                                     publicationName == "science, technology and human values" ~"science, technology\nand human values",
                                     publicationName=="science, technology and society" ~ "science, technology\nand society",
                                     publicationName=="public understanding of science" ~"public understanding\nof science",
                                     publicationName== "science and technology studies" ~"science and\ntechnology studies",
                                     TRUE ~ publicationName),
         publicationName = str_to_title(publicationName),
         publicationName = factor(publicationName, levels =str_to_title(journals_order))) %>%
  mutate(Topic= as.numeric(Topic)) %>% 
  #filter(Topic %in% as.character(c(1,3,5,7,10,11))) %>% 
  ggplot(aes(Topic,value, fill = factor(Topic), label=Topic))+
  geom_col()+
  # geom_text(position = position_dodge(width = 1),size=6, vjust=-0.1, family="LM Roman 10",
  #           data = . %>% filter(Topic %in% c(1, 4,5,6,7,9, 14,15)))+
  facet_wrap(.~publicationName, scales = 'free')+
  theme_minimal()+
  scale_x_continuous(breaks = 1:20,labels = xlabs)+
  theme(legend.position = 'none',
        text = element_text(size = 15, family="LM Roman 10"))+
  labs(x = 'Topic', y ='Relative Importance')


ggsave('results/plots/FigD1.png', dpi = 400, width = 10, height = 10)


# Fig E.1 Frobenius norm --------------------------------------------------

fn <- read_csv('results/frob_norm.csv')


fn %>% 
  ggplot(aes(citation_rank, `mean frobenious norm`, fill=citation_rank))+
  geom_col()+
  facet_wrap(.~model, scales = 'free')+
  scale_fill_gdocs()+
  theme_minimal()+
  theme(legend.position = 'none',
        text = element_text(size = 32, family="LM Roman 10"),
        axis.text.x = element_text(angle=90))+
  labs(x = 'Citation Rank', y ='Mean\nFrobenius Norm')
  

ggsave('results/plots/FigE1.png', dpi = 400, width = 12, height = 5)




