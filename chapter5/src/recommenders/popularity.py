from chapter5.src.util.models import RecommendResult, Dataset
from base_recommender import BaseRecommender
from collections import defaultdict
import numpy as np

'''
각 사용자에 대한 추천 영화는 해당 사용자가 아직 평가하지 않은 영화 중에서 평균값이 높은 10개 작품으로 한다
단, 평가 건수가 적으면 노이즈가 커지므로 minimum_num_rating건 이상 평가가 있는 영화로 한정한다
'''

np.random.seed(0)


class PopularityRecommender(BaseRecommender):
    def __init__(self):
        self.dataset = None

    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        self.dataset = dataset

        minimum_num_rating = kwargs.get("minimum_num_rating", 200)  # 평갓값의 임곗값
        movies_sorted_by_rating = self._get_movies_sorted_by_rating(minimum_num_rating)

        user_watched_movies = self.dataset.train.groupby("user_id").agg({"movie_id": list})["movie_id"].to_dict()
        unique_user_ids = sorted(self.dataset.train.user_id.unique())
        pred_user2items = defaultdict(list)

        for user_id in unique_user_ids:
            for movie_id in movies_sorted_by_rating:
                if movie_id not in user_watched_movies[user_id]:
                    pred_user2items[user_id].append(movie_id)
                if len(pred_user2items[user_id]) == 10:
                    break

        return RecommendResult(self._get_rating_pred(), pred_user2items)

    def _get_movies_sorted_by_rating(self, minimum_num_rating):
        movie_stats = self.dataset.train.groupby("movie_id").agg({"rating": ['size', 'mean']})
        atleast_flg = movie_stats["rating"]["size"] >= minimum_num_rating

        return movie_stats[atleast_flg].sort_values(by=("rating", "mean"), ascending=False).index.tolist()

    def _get_rating_pred(self):
        # 각 아이템별 평균 평갓값을 계산하고, 그 평균 평갓값을 예측값으로 사용한다
        movie_rating_average = self.dataset.train.groupby("movie_id").agg({"rating": 'mean'})

        # 테스트 데이터에 예측값을 저장한다. 테스트 데이터에만 존재하는 아이템의 예측 평갓값은 0으로 한다
        movie_rating_predict = self.dataset.test.merge(
            movie_rating_average, on="movie_id", how="left", suffixes=("_test", "_pred")
        ).fillna(0)

        return movie_rating_predict.rating_pred


if __name__ == "__main__":
    PopularityRecommender().run_sample()
