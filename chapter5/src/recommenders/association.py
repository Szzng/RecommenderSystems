from chapter5.src.util.models import RecommendResult, Dataset
from base_recommender import BaseRecommender
from collections import defaultdict, Counter
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

'''
연관 규칙을 사용해, 각 사용자가 아직 평가하지 않은 영화를 10개 추천한다
min_support와 min_threshold가 중요한 파라미터가 되므로 바꾸어 가면서 시험해보자

1. 지지도: 어떤 아이템이 전체 중에서 출현한 비율
  - 지지도 A : A의 출현 수 / 전체 데이터 수

2. 확신도: 아이템 A가 나타났을 때 아이템 B가 나타날 비율
  - 확신도 A => B = A와 B의 동시 출현 수 / A의 출현 수 (A가 조건부, B가 귀결부)
  
3. 리프트값: 아이템 A와 아이템 B의 출현이 어느 정도 상관관계를 갖는지 나타냄
  - 리프트 A => B = 지지도(A and B) / (지지도A * 지지도B)
  - 아이템 A와 아이템 B가 독립적이라면 리프트값은 1
  - 두 아이템이 양의 상관관계가 있다면 리프트값은 1보다 커짐
  - 두 아이템이 음의 상관관계가 있다면 리프트값은 1보다 작아짐
'''

np.random.seed(0)


class AssociationRecommender(BaseRecommender):
    def __init__(self):
        self.dataset = None

    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        self.dataset = dataset

        # 평갓값의 임곗값
        min_support = kwargs.get("min_support", 0.1)  # 최소 지지도 (=출현비율, 작을수록 계산 시간은 증가)
        min_lift_threshold = kwargs.get("min_threshold", 1)  # 최소 리프트값

        rules = self._get_association_rules(min_support, min_lift_threshold)

        pred_user2items = defaultdict(list)
        user_evaluated_movies = dataset.train.groupby("user_id").agg({"movie_id": list})["movie_id"].to_dict()
        movielens_train_high_rating = dataset.train[dataset.train.rating >= 4]  # 학습용 데이터에서 평갓값이 4 이상인 데이터만 추출

        for user_id, data in movielens_train_high_rating.groupby("user_id"):
            # 사용자가 직전에 평가한 5개의 영화를 얻는다
            input_data = data.sort_values("timestamp")["movie_id"].tolist()[-5:]
            # 그 영화들이 조건부에 하나라도 포함되는 연관 규칙을 검출한다
            matched_flags = rules.antecedents.apply(lambda x: len(set(input_data) & x)) >= 1

            # 연관 규칙의 귀결부의 영화를 리스트에 저장하고, 등록 빈도 수로 정렬해 사용자가 아직 평가하지 않았다면, 추천 목록에 추가한다
            consequent_movies = []
            for i, row in rules[matched_flags].sort_values("lift", ascending=False).iterrows():
                consequent_movies.extend(row["consequents"])

            # 등록 빈도 세기
            counter = Counter(consequent_movies)
            for movie_id, movie_cnt in counter.most_common():
                if movie_id not in user_evaluated_movies[user_id]:
                    pred_user2items[user_id].append(movie_id)

                if len(pred_user2items[user_id]) == 10:
                    break

        # 연관 규칙에서는 평갓값을 예측하지 않으므로, rmse 평가는 수행하지 않는다(편의상, 테스트 데이터의 예측값을 그대로 반환).
        return RecommendResult(dataset.test.rating, pred_user2items)

    def _get_association_rules(self, min_support, min_lift_threshold):
        # 사용자 x 영화 행렬 형식으로 변경
        user_movie_matrix = self.dataset.train.pivot(index="user_id", columns="movie_id", values="rating")
        # 라이브러리 사용을 위해 4 이상의 평갓값은 True, 4 미만의 평갓값은 False으로 한다
        user_movie_matrix = user_movie_matrix >= 4

        # 지지도가 높은 영화
        freq_movies = apriori(user_movie_matrix, min_support=min_support, use_colnames=True)

        # association_rules: 빈번한 아이템셋(frequent itemsets)을 입력으로 받아 연관 규칙을 계산
        # 리프트값이 높은 순으로 표시, 리프트값이 1 이하인 규칙은 제외
        rules = association_rules(freq_movies, metric="lift", min_threshold=min_lift_threshold)

        return rules


if __name__ == "__main__":
    AssociationRecommender().run_sample()
