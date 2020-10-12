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
# Fig. 1 degree distribution -----------------------------------------------------

degree <- read_csv('.../results/network_degree.csv')


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

# Fig. 3/4. TSNE_projection ---------------------------------------------------------
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
  theme_void()+
  # scale_color_gdocs()+
  scale_color_manual(values = colors)+
  # scale_color_viridis_d()+
  # scale_alpha(range = c(0.5,1))+
  scale_size_continuous(range = c(0.001,10))+
  theme(text = element_text(size = 36, family="LM Roman 10"),
        legend.position = 'bottom',legend.box="vertical")+
  guides(color=guide_legend(override.aes = list(size=20),nrow = 3,title = 'Journal'), size = guide_legend(title = 'Citations'), alpha= guide_legend(title = 'Citations'))

ggsave('results/plots/Fig3.png', dpi = 300, width = 10, height = 10, scale = 2)


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

p2 <- df_semantic %>% 
  filter(model %in% c('BERT')) %>% 
  ggplot(.,aes(x,y,color=publicationName, size=citedby_count, alpha=citedby_count))+
  geom_point()+
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

ggsave(plot = g,'results/plots/Fig4.png', dpi = 300, width = 10, height = 15, scale = 2)


# Fig 5. Heatmaps. cosine similarity by collaboration status ----------------------

av_dist_collaboration_df_gnn <- read_csv("results/av_dist_collaboration_df_gnn.csv")
av_dist_collaboration_df_bert <- read_csv("results/av_dist_collaboration_df_bert.csv")
av_dist_collaboration_df_tm <- read_csv("results/av_dist_collaboration_df_tm.csv")

heatmap_collab <- function(data,text_size=24, legend_position='right' ){

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
        plot.margin=grid::unit(c(0,0,0,0), "mm"))


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
ggsave(glue('results/plots/Fig5a.png'), dpi=400, width = 10, height = 8)

heatmap_collab(av_dist_collaboration_df_tm, text_size = 31,legend_position = 'left')
ggsave(glue('results/plots/Fig5b.png'), dpi=400, width = 10, height = 8)

heatmap_collab(av_dist_collaboration_df_bert, text_size = 31, legend_position = 'right')
ggsave(glue('results/plots/Fig5c.png'), dpi=400, width = 10, height = 8)



#Fig. 6 Countries average ----------------------------------------------

  
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
  

ggsave('results/plots/Fig6.png', dpi = 400, width = 10, height = 7, scale = 2)


# Fig. 7 avg_embed_journal -------------------------------------------------------
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
                           field == 'Education, Communication and Anthropology' ~ 'Education, Communication\nand Anthropology',
                           field == 'Library and Information Sciences' ~ 'Library and\nInformation Sciences',
                           TRUE ~field),
         mean_citations = n_citations/n)

colors = c("#351ced","#d12f06","#008709","#990099")

fields <- c("Library and\nInformation Sciences", "Management", "Education, Communication\nand Anthropology", "History and Philosophy")
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
  labs(x = 'Qualitative-Quantitateive axis', y ='', color ='Field', size= 'Mean citations', shape = 'Field')+
  guides(color=guide_legend(override.aes = list(size=10),nrow=2), size = guide_legend(), shape = guide_legend())+
  facet_wrap(.~type, scales = 'free')
  
ggsave('results/plots/Fig7.png', dpi = 400, width = 18, height = 12)


# Fid D.1 Frobenius norm --------------------------------------------------

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
  

ggsave('results/plots/FigD1.png', dpi = 400, width = 12, height = 5)


