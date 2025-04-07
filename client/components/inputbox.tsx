import { Input } from "@/components/ui/input";

interface InputBoxProps {
  placeholder: string;
  onChange: (value: string) => void;
}

export default function InputBox({placeholder, onChange }: InputBoxProps) {
  return (
    <div>
        <Input
          type="learning_rate"
          placeholder={placeholder}
          onChange={(e) => onChange(e.target.value)}
        />
    </div>
   
  );
}
