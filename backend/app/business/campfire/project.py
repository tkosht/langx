from collections import namedtuple

ProjectRecord = namedtuple(
    "ProjectRecord",
    [
        "project_id",
        "title",
        "project_url",
        "data_dimension",
        "data_brand",
        "data_category",
        "page",
        "data_position",
        "data_list",
        "img_url",
        "prefecture",
        "project_type",
        "user_id",
        "success_rate",
        "current_funding",
        "current_supporters",
        "remaining_days",
        "member",
        "status",
    ],
)


ProjectDetails = namedtuple(
    "ProjectDetails",
    [
        "type",
        "project_url",
        "title",
        "img_url",
        "backer_amount",
        "abstract",
        "article_text",
        "article_html",
        "profile_text",
        "profile_url",
        "icon_url",
        "user_name",
        "prefecture",
        "project_exprience",
        "readmore",
        "return_boxes",
    ],
)

ReturnBox = namedtuple("ReturnBox", ["return_idx", "return_img_url", "price", "desc"])
