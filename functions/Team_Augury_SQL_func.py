import pandas as pd
### SQL downloads


def sql_by_timestamp(conn,sr_id,lower_timestamp,upper_timestamp):

    ### POST

    sql = '''
        SELECT pot.batch_id, b.time_of_batch, pot.post_id, pot.hours_since_created, p.sr, p.post_text, pot.upvotes AS post_upvotes, pot.number_comments, p.created_at AS post_created_at, 
            r.author_id
        FROM post_over_time AS pot
        INNER JOIN post AS p ON p.id = pot.post_id
        INNER JOIN batch AS b ON b.id = pot.batch_id
        INNER JOIN redditor AS r on r.id = p.author_id
        WHERE p.sr IN {} AND b.time_of_batch >= '{}'::timestamp AND b.time_of_batch <= '{}'::timestamp 

        '''.format(sr_id, lower_timestamp, upper_timestamp)

    post_data = pd.read_sql_query(sql, conn)                
    #display(post_data)

    ### then get df of Author karma over time (haven't been able to do an efficient JOIN in SQL)
    sql = '''
        SELECT rot.batch_id,   
            r.author_id ,rot.karma

        FROM redditor_over_time AS rot
        INNER JOIN redditor AS r on r.id = rot.author_id
        INNER JOIN post AS p ON p.author_id = r.id
        INNER JOIN batch AS b ON b.id = rot.batch_id

        WHERE p.sr IN {} AND b.time_of_batch >= '{}'::timestamp AND b.time_of_batch <= '{}'::timestamp  

        '''.format(sr_id, lower_timestamp, upper_timestamp)

    post_karma_history = pd.read_sql_query(sql, conn)
    #display(post_karma_history)

    sql = '''
        SELECT rot.batch_id,   
            r.author_id ,rot.karma

        FROM redditor_over_time AS rot
        INNER JOIN redditor AS r on r.id = rot.author_id
        INNER JOIN comments AS c ON c.author_id = r.id
        INNER JOIN post AS p ON p.id = c.post_id
        INNER JOIN batch AS b ON b.id = rot.batch_id

        WHERE p.sr IN {} AND b.time_of_batch >= '{}'::timestamp AND b.time_of_batch <= '{}'::timestamp  

        '''.format(sr_id, lower_timestamp, upper_timestamp)

    comments_karma_history = pd.read_sql_query(sql, conn)
    #display(post_karma_history)


    ### Comments
    sql = '''
        SELECT cot.batch_id, b.time_of_batch, p.id AS post_id, p.sr, cot.comment_id, c.comment_text, cot.upvotes AS comment_upvotes, c.created_at AS comment_created_at,
            r.author_id

        FROM comments_over_time AS cot
        INNER JOIN comments AS c ON c.id = cot.comment_id
        INNER JOIN batch AS b ON b.id = cot.batch_id    
        INNER JOIN post AS p ON p.id = c.post_id
        INNER JOIN redditor AS r on r.id = c.author_id

        WHERE p.sr IN {} AND b.time_of_batch >= '{}'::timestamp AND b.time_of_batch <= '{}'::timestamp 

        '''.format(sr_id, lower_timestamp, upper_timestamp)

    comments_data = pd.read_sql_query(sql, conn)
    #display(comments_data)


    ### merge the karma df with posts and comments
    ### also merge hours_since_created to comments
    ### duplicate batch_id / author_id occurs, my suspicion is because an author of a post then is also the author of a comment in the same batch.
    ### we fix this with .drop_duplicates()
    ### also drop any duplicate hours_since_created, if there are any

    post_data = post_data.merge(post_karma_history.drop_duplicates(subset=['batch_id','author_id']), how='left', on=['batch_id', 'author_id', ]).rename({'karma':'post_author_karma'},axis=1).drop(['author_id'],axis=1)
    comments_data = comments_data.merge(comments_karma_history.drop_duplicates(subset=['batch_id','author_id']), how='left', on=['batch_id', 'author_id', ]).rename({'karma':'comment_author_karma'},axis=1).drop(['author_id'],axis=1)
    comments_data = comments_data.merge(post_data[['batch_id','post_id','hours_since_created']].copy(), how='left', on=['batch_id', 'post_id'])

    post_data = post_data.drop_duplicates(subset=['post_id','hours_since_created'])
    comments_data = comments_data.drop_duplicates(subset=['comment_id','hours_since_created'])

    #print('-- raw posts --')
    #display(post_data)
    #print('-- raw comments --' )
    #display(comments_data)

    del post_karma_history, comments_karma_history #remove un-needed df's

    return post_data, comments_data