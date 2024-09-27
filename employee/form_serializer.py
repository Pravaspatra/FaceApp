from rest_framework import serializers
from employee.models import EmployeePersonalInfo
from company.models import CompanyMeta, CompanyContactInfo, CompanyBranchInfo
from attendance.models import EmployeeTimeSheet

class EmployeePersonalInfoSerializer(serializers.ModelSerializer):
    class Meta:
        model = EmployeePersonalInfo
        fields = ['gender', 'aadhar', 'mobile_number']

class EmployeePersonalInfoSerializer(serializers.ModelSerializer):
    class Meta:
        model = EmployeeTimeSheet
        fields = ['details']
