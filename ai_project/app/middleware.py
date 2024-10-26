import logging
import traceback
from django.http import JsonResponse

logger = logging.getLogger(__name__)

class ErrorLoggingMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        try:
            response = self.get_response(request)
            return response
        except Exception as e:
            logger.error(f"Unhandled exception: {str(e)}")
            logger.error(traceback.format_exc())
            return JsonResponse(
                {'error': 'Internal server error', 'detail': str(e)}, 
                status=500
            )

    def process_exception(self, request, exception):
        logger.error(f"Exception in request: {str(exception)}")
        logger.error(f"Request data: {request.POST or request.GET}")
        logger.error(traceback.format_exc())
        return JsonResponse(
            {'error': 'Internal server error', 'detail': str(exception)}, 
            status=500
        )