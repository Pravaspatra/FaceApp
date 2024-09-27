from rest_framework import serializers
from employee.models import EmployeeDesignation, EmployeeCompanyInfo, EmployeeDocumentLocker, EmployeePersonalInfo, EmployeeFinancialInfo
from company.models import CompanyMeta, CompanyContactInfo, CompanyBranchInfo
from attendance.models import EmployeeShiftAssociation,EmployeeAttendanceSetting, EmployeeTimeSheet

class DesignationDropdownSerializer(serializers.ModelSerializer):
    class Meta:
        model = EmployeeDesignation
        fields = ['id', 'name', 'is_admin']


class EmployeeDocumentsSerializer(serializers.ModelSerializer):

    name = serializers.SerializerMethodField()
    attachments = serializers.SerializerMethodField()

    class Meta:
        model = EmployeeDocumentLocker
        fields = ['name', 'attachments']


    def get_name(self, obj):
        return obj.group.name

    def get_attachments(self, obj):
        urls = []
        photos = obj.photos.all()
        print("photos.count()")        
        print(photos.count())
        for photo in photos:
            photo =  photo.photo
            if photo and hasattr(photo, 'url'):
                urls.append(photo.url)
        return urls


class EmployeeCompanyInfoSerializer(serializers.ModelSerializer):

    name = serializers.SerializerMethodField()
    is_admin = serializers.SerializerMethodField()
    is_branch_admin = serializers.SerializerMethodField()
    is_active = serializers.SerializerMethodField()
    mobile_number = serializers.SerializerMethodField()
    can_reregister_face = serializers.SerializerMethodField()
    department_id = serializers.SerializerMethodField()
    designation_id = serializers.SerializerMethodField()

    class Meta:
        model = EmployeeCompanyInfo
        fields = ['id', 'mobile_number', 'employee_id', 'name', 'is_admin', 'is_active', 'is_branch_admin', 'can_reregister_face', 'department_id', 'designation_id']

    def get_department_id(self, obj):      
        return obj.department.id

    def get_designation_id(self, obj):      
        return obj.designation.id

    def get_can_reregister_face(self, obj):      
        return True

    def get_mobile_number(self, obj):
        return EmployeePersonalInfo.objects.get(user = obj.user).mobile_number

    def get_name(self, obj):
        return obj.user.first_name

    def get_is_admin(self, obj):
        return False

    def get_is_branch_admin(self, obj):
        return False

    def get_is_active(self, obj):
        return True


class EmployeesWitShiftSerializer(serializers.ModelSerializer):

    name = serializers.SerializerMethodField()
    is_admin = serializers.SerializerMethodField()
    is_branch_admin = serializers.SerializerMethodField()
    is_active = serializers.SerializerMethodField()
    mobile_number = serializers.SerializerMethodField()
    can_reregister_face = serializers.SerializerMethodField()
    department_id = serializers.SerializerMethodField()
    designation_id = serializers.SerializerMethodField()
    shift = serializers.SerializerMethodField()

    class Meta:
        model = EmployeeCompanyInfo
        fields = ['id', 'mobile_number', 'employee_id', 'name', 'is_admin', 'is_active', 'is_branch_admin', 'can_reregister_face', 'department_id', 'designation_id', 'shift']

    def get_department_id(self, obj):      
        return obj.department.id

    def get_designation_id(self, obj):      
        return obj.designation.id

    def get_can_reregister_face(self, obj):      
        return True

    def get_mobile_number(self, obj):
        return EmployeePersonalInfo.objects.get(user = obj.user).mobile_number

    def get_name(self, obj):
        return obj.user.first_name

    def get_is_admin(self, obj):
        return False

    def get_is_branch_admin(self, obj):
        return False

    def get_is_active(self, obj):
        return True

    def get_shift(self, obj):
        try:
            employeeShiftAssociation = EmployeeShiftAssociation.objects.get(associated_user = obj)
            return {'id': employeeShiftAssociation.shift_default.id, 'name':employeeShiftAssociation.shift_default.name}
        except Exception as e:
            print("eeeeeeee", e)
            return None
class EmployeesDocumentsSerializer(serializers.ModelSerializer):

    name = serializers.SerializerMethodField()
    is_admin = serializers.SerializerMethodField()
    is_branch_admin = serializers.SerializerMethodField()
    is_active = serializers.SerializerMethodField()
    mobile_number = serializers.SerializerMethodField()
    count = serializers.SerializerMethodField()

    class Meta:
        model = EmployeeCompanyInfo
        fields = ['id', 'mobile_number', 'employee_id', 'name', 'is_admin', 'is_active', 'is_branch_admin', 'count']


    def get_mobile_number(self, obj):
        return EmployeePersonalInfo.objects.get(user = obj.user).mobile_number
        # return obj.user.first_name

    def get_name(self, obj):
        return obj.user.first_name

    def get_is_admin(self, obj):
        return False

    def get_is_branch_admin(self, obj):
        return False

    def get_is_active(self, obj):
        return True

    def get_count(self, obj):
        return 10

        


class EmployeeCompanyInfoSerializerWeb(serializers.ModelSerializer):

    name = serializers.SerializerMethodField()
    mobile_number = serializers.SerializerMethodField()
    branch = serializers.SerializerMethodField()
    department_id = serializers.SerializerMethodField()
    designation_id = serializers.SerializerMethodField()
    shift = serializers.SerializerMethodField()


    class Meta:
        model = EmployeeCompanyInfo
        fields = ['id', 'mobile_number', 'employee_id', 'name', 'branch', 'department_id', 'designation_id', 'shift']

    def get_department_id(self, obj):      
        return obj.department.id

    def get_designation_id(self, obj):      
        return obj.designation.id

    def get_mobile_number(self, obj):
        return EmployeePersonalInfo.objects.get(user = obj.user).mobile_number

    def get_name(self, obj):
        return obj.user.first_name

    def get_branch(self, obj):
        return obj.company_branch.name

    def get_shift(self, obj):
        try:
            employeeShiftAssociation = EmployeeShiftAssociation.objects.get(associated_user = obj)
            return {'id': employeeShiftAssociation.shift_default.id, 'name':employeeShiftAssociation.shift_default.name}
        except:
            return None

class EmployeesTimeSheetsSerializer(serializers.ModelSerializer):

    name = serializers.SerializerMethodField()
    is_admin = serializers.SerializerMethodField()
    is_branch_admin = serializers.SerializerMethodField()
    is_active = serializers.SerializerMethodField()
    mobile_number = serializers.SerializerMethodField()
    timesheet_entries_count = serializers.SerializerMethodField()
    user_id = serializers.SerializerMethodField()

    
    class Meta:
        model = EmployeeCompanyInfo
        fields = ['id', 'user_id', 'mobile_number', 'employee_id', 'name', 'is_admin', 'is_active', 'is_branch_admin', 'timesheet_entries_count']

    def get_user_id(self, obj):
        return obj.user.id
        
    def get_mobile_number(self, obj):
        return EmployeePersonalInfo.objects.get(user = obj.user).mobile_number

    def get_name(self, obj):
        return obj.user.first_name

    def get_timesheet_entries_count(self, obj):
        return 0
        # return EmployeeTimeSheet.objects.filter(user = obj.user).count()

    def get_is_admin(self, obj):
        return False

    def get_is_branch_admin(self, obj):
        return False

    def get_is_active(self, obj):
        return True

class EmployeeDetailsSerializer(serializers.ModelSerializer):

    is_admin = serializers.SerializerMethodField()
    is_branch_admin = serializers.SerializerMethodField()
    is_active = serializers.SerializerMethodField()
    mobile_number = serializers.SerializerMethodField()
    first_name = serializers.SerializerMethodField()
    last_name = serializers.SerializerMethodField()
    email = serializers.SerializerMethodField()
    gender = serializers.SerializerMethodField()
    pan = serializers.SerializerMethodField()
    aadhar_number = serializers.SerializerMethodField()
    designation_id = serializers.SerializerMethodField()
    department_id = serializers.SerializerMethodField()
    branch_id = serializers.SerializerMethodField()
    attendance_settings = serializers.SerializerMethodField()
    dob = serializers.SerializerMethodField()
    blood_group = serializers.SerializerMethodField()
    shift = serializers.SerializerMethodField()

    class Meta:
        model = EmployeeCompanyInfo
        fields = ['id', 'mobile_number', 'dob', 'blood_group', 'date_of_joining', 'employment_type', 'kgid_number', 'attendance_settings', 'department_id', 'branch_id', 'designation_id', 'gender', 'designation_id', 'aadhar_number', 'pan', 'email', 'is_admin', 'is_active', 'is_branch_admin', 'first_name', 'last_name', 'shift']


    def get_blood_group(self, obj):
        return  EmployeePersonalInfo.objects.get(user = obj.user).blood_group

    def get_dob(self, obj):
        return  EmployeePersonalInfo.objects.get(user = obj.user).dob

    def get_designation_id(self, obj):
        return obj.designation.id

    def get_department_id(self, obj):
        return obj.department.id

    def get_attendance_settings(self, obj):
        employeeAttendanceSetting = EmployeeAttendanceSetting.objects.get(user = obj.user)
        response = {}
        response['start_time'] = employeeAttendanceSetting.start_time
        response['end_time'] = employeeAttendanceSetting.end_time
        response['break_start_time'] = employeeAttendanceSetting.break_start_time
        response['break_end_time'] = employeeAttendanceSetting.break_end_time
        response['id'] = employeeAttendanceSetting.id
        response['is_excempt_allowed'] = employeeAttendanceSetting.is_excempt_allowed
        response['can_field_checkin'] = employeeAttendanceSetting.can_field_checkin
        response['can_office_checkin'] = employeeAttendanceSetting.can_office_checkin
        response['face_validation_required'] = employeeAttendanceSetting.face_validation_required
        return response

    def get_shift(self, obj):
        try:
            employeeShiftAssociation = EmployeeShiftAssociation.objects.get(associated_user = obj)
            return {'id': employeeShiftAssociation.shift_default.id, 'name':employeeShiftAssociation.shift_default.name}
        except:
            return None

    def get_branch_id(self, obj):
        return obj.company_branch.id

    def get_gender(self, obj):
        return  EmployeePersonalInfo.objects.get(user = obj.user).gender

    def get_pan(self, obj):
        return EmployeeFinancialInfo.objects.get(user = obj.user).pan

    def get_aadhar_number(self, obj):
        return EmployeePersonalInfo.objects.get(user = obj.user).aadhar

    def get_first_name(self, obj):
        return obj.user.first_name

    def get_last_name(self, obj):
        return obj.user.last_name

    def get_email(self, obj):
        return obj.user.email
        # obj.user.email

    def get_mobile_number(self, obj):
        return EmployeePersonalInfo.objects.get(user = obj.user).mobile_number

    def get_is_admin(self, obj):
        return False

    def get_is_branch_admin(self, obj):
        return False

    def get_is_active(self, obj):
        return True
