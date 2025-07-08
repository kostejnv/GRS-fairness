import numpy as np
import pandas as pd

from abc import ABC, abstractmethod

class AggregationStrategy(ABC):
    name: str
    
    def __init__(self, name: str):
        self.name = name

    @staticmethod
    def getAggregator(strategy):
        if strategy == "ADD":
            return BaselinesAggregator("ADD")
        elif strategy == "MUL":
            return BaselinesAggregator("MUL")
        elif strategy == "LMS":
            return BaselinesAggregator("LMS")
        elif strategy == "MPL":
            return BaselinesAggregator("MPL")
        elif strategy == "GFAR":
            return GFARAggregator()
        elif strategy == "EPFuzzDA":
            return EPFuzzDAAggregator()
        else:
            raise ValueError(f"Strategy {strategy} not supported. Check typos.")

    @abstractmethod
    def generate_group_recommendations_for_group(self, group_ratings, recommendations_number) -> dict[str, list]:
        ...


class BaselinesAggregator(AggregationStrategy):
    def __init__(self, aggregation_strategy):
        assert aggregation_strategy in ["ADD", "MUL", "LMS", "MPL"], "Invalid aggregation strategy"
        super().__init__(aggregation_strategy)
    
    def generate_group_recommendations_for_group(self, group_ratings, recommendations_number):
        if self.name == "ADD":
            agg_df = group_ratings.groupby('item')['predicted_rating'].sum().reset_index(name='score')
        elif self.name == "MUL":
            agg_df = group_ratings.groupby('item')['predicted_rating'].prod().reset_index(name='score')
        elif self.name == "LMS":
            agg_df = group_ratings.groupby('item')['predicted_rating'].min().reset_index(name='score')
        elif self.name == "MPL":
            agg_df = group_ratings.groupby('item')['predicted_rating'].max().reset_index(name='score')
        else:
            raise ValueError(f"Unknown aggregation strategy: {self.name}")

        recommendation_list = list(
            agg_df.sort_values(by="score", ascending=False).head(recommendations_number)['item']
        )
        return {self.name: recommendation_list}


from scipy.stats import rankdata
import numpy as np

class GFARAggregator(AggregationStrategy):
    def __init__(self):
        super().__init__("GFAR")

    # ---------- helpers --------------------------------------------------
    @staticmethod
    def _top_k(df, k):
        """Return k rows with highest predicted_rating."""
        return df.nlargest(k, 'predicted_rating')

    @staticmethod
    def _borda_scores(top_df):
        """Highest rating ⇒ highest Borda weight."""
        k = len(top_df)
        # rankdata on the *negative* turns descending ordering into ranks 1…k
        return k - rankdata(-top_df["predicted_rating"].values, method="ordinal") + 1

    # ---------- core GFAR ------------------------------------------------
    def gfar_algorithm(self, ratings, top_n, max_rel_items=20):
        df = ratings.copy()
        df["p_rel"] = 0.0              # P(item relevant for this user)
        df["p_none"] = 1.0             # P(user still unsatisfied)

        users = df.user.unique()

        # (1) Per-user relevance probabilities
        for u in users:
            sub = df[df.user == u]
            top = self._top_k(sub, max_rel_items)
            borda = self._borda_scores(top)
            df.loc[top.index, "p_rel"] = borda / borda.sum()   # normalise

        slate = []
        for _ in range(top_n):
            # (2) Marginal gain of each candidate item
            df["gain"] = df.p_rel * df.p_none
            gains = df.groupby("item")["gain"].sum()

            item_id = gains.idxmax()          # **fixed** (was .argmax())
            slate.append(item_id)

            # (3) Update P(user still none)
            hits = df[df.item == item_id]
            for uid, p in zip(hits.user, hits.p_rel):
                df.loc[df.user == uid, "p_none"] *= (1 - p)

            # (4) Remove chosen item
            df = df[df.item != item_id]

        return slate

    def generate_group_recommendations_for_group(self, group_ratings, k):
        return {"GFAR": self.gfar_algorithm(group_ratings, k)}


class EPFuzzDAAggregator(AggregationStrategy):
    # implements EP-FuzzDA aggregation algorithm. For more details visit https://dl.acm.org/doi/10.1145/3450614.3461679
    def __init__(self):
        super().__init__("EPFuzzDA")

    def ep_fuzzdhondt_algorithm(self, group_ratings, top_n, member_weights=None):
        group_members = group_ratings.user.unique()
        all_items = group_ratings["item"].unique()
        group_size = len(group_members)

        if not member_weights:
            member_weights = [1. / group_size] * group_size
        member_weights = pd.DataFrame(pd.Series(member_weights, index=group_members))

        localDF = group_ratings.copy()

        candidate_utility = pd.pivot_table(localDF, values="predicted_rating", index="item", columns="user",
                                           fill_value=0.0)
        candidate_sum_utility = pd.DataFrame(candidate_utility.sum(axis="columns"))

        total_user_utility_awarded = pd.Series(np.zeros(group_size), index=group_members)
        total_utility_awarded = 0.

        selected_items = []
        # top-n times select one item to the final list
        for i in range(top_n):
            # print()
            # print('Selecting item {}'.format(i))
            # print('Total utility awarded: ', total_utility_awarded)
            # print('Total user utility awarded: ', total_user_utility_awarded)

            prospected_total_utility = candidate_sum_utility + total_utility_awarded  # pd.DataFrame items x 1

            # print(prospected_total_utility.shape, member_weights.T.shape)

            allowed_utility_for_users = pd.DataFrame(np.dot(prospected_total_utility.values, member_weights.T.values),
                                                     columns=member_weights.T.columns,
                                                     index=prospected_total_utility.index)

            # print(allowed_utility_for_users.shape)

            # cap the item's utility by the already assigned utility per user
            unfulfilled_utility_for_users = allowed_utility_for_users.subtract(total_user_utility_awarded,
                                                                               axis="columns")
            unfulfilled_utility_for_users[unfulfilled_utility_for_users < 0] = 0

            candidate_user_relevance = pd.DataFrame(np.minimum(unfulfilled_utility_for_users, candidate_utility))
            candidate_relevance = candidate_user_relevance.sum(axis="columns")

            # remove already selected items
            candidate_relevance = candidate_relevance.loc[~candidate_relevance.index.isin(selected_items)]
            item_pos = candidate_relevance.argmax()
            item_id = candidate_relevance.index[item_pos]

            # print(item_pos,item_id,candidate_relevance[item_id])

            # print(candidate_relevance.index.difference(candidate_utility.index))
            # print(item_id in candidate_relevance.index, item_id in candidate_utility.index)
            selected_items.append(item_id)

            winner_row = candidate_utility.loc[item_id, :]
            # print(winner_row)
            # print(winner_row.shape)
            # print(item_id,item_pos,candidate_relevance.max())
            # print(selected_items)
            # print(total_user_utility_awarded)
            # print(winner_row.iloc[0,:])

            total_user_utility_awarded.loc[:] = total_user_utility_awarded.loc[:] + winner_row

            total_utility_awarded += winner_row.values.sum()
            # print(total_user_utility_awarded)
            # print(total_utility_awarded)

        return selected_items

    def generate_group_recommendations_for_group(self, group_ratings, recommendations_number):
        selected_items = self.ep_fuzzdhondt_algorithm(group_ratings, recommendations_number)
        return {"EPFuzzDA": selected_items}