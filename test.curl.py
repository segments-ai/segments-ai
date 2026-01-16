curl -X GET \
    'http://localhost:8000/datasets/admin-antoine-local/test-image/samples/?per_page=500&page=1&include_full_label=1&sort=name' \
    -H 'Authorization: APIKey ac459f8c635ec615cf51df1f0a9abb3ec1109798' \
    -H 'X-source: python-sdk' \
    -H 'Segments-SDK-Version: 1.19.1'
