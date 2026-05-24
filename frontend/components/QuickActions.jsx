import { Calendar, Plane, CreditCard } from "lucide-react";
import { useLanguage } from "@/contexts/LanguageContext";

export default function QuickActions({ onActionClick }) {
  const { t } = useLanguage();

  // 자주 묻는 질문(Quick Actions) 데이터
  const quickActions = [
    { label: t("quickActions.academicCalendar"), icon: <Calendar size={14} />, query: t("quickActions.academicCalendar"), payload: "QA_1" },
    { label: t("quickActions.exchangeStudent"), icon: <Plane size={14} />, query: t("quickActions.exchangeStudent"), payload: "QA_2" },
    { label: t("quickActions.tuition"), icon: <CreditCard size={14} />, query: t("quickActions.tuition"), payload: "QA_3" },
  ];

  return (
    <div className="flex gap-2 px-4 pb-2 overflow-x-auto scrollbar-hide items-center">
      {quickActions.map((action, idx) => (
        <button
          key={idx}
          onClick={() => onActionClick(action.query, action.payload)}
          className="flex items-center gap-1.5 whitespace-nowrap bg-white text-[#003876] py-1.5 px-3 rounded-full text-xs font-medium border border-transparent hover:bg-[#003876] hover:text-white hover:border-white active:bg-[#003876] active:text-white active:border-white transition-colors"
        >
          {action.icon}
          {action.label}
        </button>
      ))}
    </div>
  );
}
