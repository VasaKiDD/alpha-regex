import numpy as np


class RegexGenerator:
    def __init__(self, num_bytes=256):
        """
        Regex generator environment for generating regex sequentially
        :param num_bytes: 128 or 256 depending if you analyze bytes
        """
        self.controls = [b"*", b"+", b"|", b"?"]  # 0-10
        self.char_sets = [b".", b"\d", b"\D", b"\w", b"\W", b"\s", b"\S"]  # 11-16 b".*?", b".*"
        self.programs = ["bracket", "parenthesis", "curly_bracket"]
        byte_chars = [bytes([i]) for i in range(num_bytes)]  # 17-272
        byte_chars[36] = b"\x5c\x24"
        byte_chars[40] = b"\x5c\x28"
        byte_chars[41] = b"\x5c\x29"
        byte_chars[42] = b"\x5c\x2a"
        byte_chars[43] = b"\x5c\x2b"
        byte_chars[46] = b"\x5c\x2e"
        byte_chars[63] = b"\x5c\x3f"
        byte_chars[91] = b"\x5c\x5b"
        byte_chars[92] = b"\x5c\x5c"
        byte_chars[93] = b"\x5c\x5d"
        byte_chars[94] = b"\x5c\x5e"
        byte_chars[123] = b"\x5c\x7b"
        byte_chars[124] = b"\x5c\x7c"
        byte_chars[125] = b"\x5c\x7d"
        self.byte_chars = byte_chars  # [32:]
        self.end = ["return"]

        self.action_space = (
            len(self.controls)
            + len(self.char_sets)
            + len(self.programs)
            + len(self.byte_chars)
            + 1
        )
        self.program_stack = [["main", []]]
        self.remaining_bracket_mask = np.ones(len(self.byte_chars))
        self.curly_values = []

    def reset(self):

        controls_mask = np.zeros(len(self.controls))
        char_sets_mask = np.ones(len(self.char_sets))
        programs_mask = np.ones(len(self.programs))
        programs_mask[-1] = 0.0
        byte_chars_mask = np.ones(len(self.byte_chars))
        end_mask = np.zeros(1)
        self.program_stack = [["main", []]]
        self.remaining_bracket_mask = np.ones(len(self.byte_chars))
        self.curly_values = []

        return np.concatenate(
            (controls_mask, char_sets_mask, programs_mask, byte_chars_mask, end_mask)
        )

    def reset_with_state(self):

        controls_mask = np.zeros(len(self.controls))
        char_sets_mask = np.ones(len(self.char_sets))
        programs_mask = np.ones(len(self.programs))
        programs_mask[-1] = 0.0
        byte_chars_mask = np.ones(len(self.byte_chars))
        end_mask = np.zeros(1)
        self.program_stack = [["main", []]]
        self.remaining_bracket_mask = np.ones(len(self.byte_chars))
        self.curly_values = []

        return (
            np.concatenate(
                (controls_mask, char_sets_mask, programs_mask, byte_chars_mask, end_mask)
            ),
            (self.program_stack, self.remaining_bracket_mask, self.curly_values),
        )

    def step(self, action):

        if self.program_stack[-1][0] in ["main", "parenthesis"]:
            return self.char_mode(action)
        if self.program_stack[-1][0] == "bracket":
            return self.bracket_mode(action)
        if self.program_stack[-1][0] == "curly_bracket":
            return self.curly_bracket_mode(action)

    def step_with_state(self, action, state):
        self.program_stack, self.remaining_bracket_mask, self.curly_values = state

        new_mask = self.step(action)

        self.close_trace()
        reg = self.generate_regex(self.program_stack[0][1])

        return new_mask, reg, (self.program_stack, self.remaining_bracket_mask, self.curly_values)

    def get_regex(self, state):
        self.program_stack, self.remaining_bracket_mask, self.curly_values = state

        self.close_trace()
        reg = self.generate_regex(self.program_stack[0][1])

        return reg

    def char_mode(self, action):

        if action < len(self.controls):
            action = self.controls[action]
            self.program_stack[-1][1].append(action)
            action_type = "control"
        elif action < len(self.controls) + len(self.char_sets):
            action = self.char_sets[action - len(self.controls)]
            self.program_stack[-1][1].append(action)
            action_type = "char_set"
        elif action < len(self.controls) + len(self.char_sets) + len(self.programs):
            action = self.programs[action - len(self.controls) - len(self.char_sets)]
            self.program_stack.append([action, []])
            action_type = "open_program"
        elif action < len(self.controls) + len(self.char_sets) + len(self.programs) + len(
            self.byte_chars
        ):
            action = self.byte_chars[
                action - len(self.controls) - len(self.char_sets) - len(self.programs)
            ]
            self.program_stack[-1][1].append(action)
            action_type = "char"
        else:
            action_type = "return"
            if self.program_stack[-1][0] == "main":
                return None
            if self.program_stack[-1][0] == "parenthesis":
                self.program_stack[-1][1].insert(0, b"(")
                self.program_stack[-1][1].append(b")")
                self.program_stack[-2][1].append(self.program_stack[-1][1])
                # print(self.generate_regex(self.program_stack[-1][1]))
                self.program_stack.pop()

        # print(action_type)
        if action_type == "open_program":
            if action == "parenthesis":
                controls_mask = np.zeros(len(self.controls))
                char_sets_mask = np.ones(len(self.char_sets))
                programs_mask = np.ones(len(self.programs))
                programs_mask[-1] = 0.0
                byte_chars_mask = np.ones(len(self.byte_chars))
                end_mask = np.zeros(1)
                return np.concatenate(
                    (controls_mask, char_sets_mask, programs_mask, byte_chars_mask, end_mask)
                )
            elif action == "bracket":
                controls_mask = np.zeros(len(self.controls))
                char_sets_mask = np.zeros(len(self.char_sets))
                programs_mask = np.zeros(len(self.programs))
                self.remaining_bracket_mask = np.ones(len(self.byte_chars))
                end_mask = np.zeros(1)
                return np.concatenate(
                    (
                        controls_mask,
                        char_sets_mask,
                        programs_mask,
                        self.remaining_bracket_mask,
                        end_mask,
                    )
                )
            elif action == "curly_bracket":
                self.curly_values = []
                controls_mask = np.ones(len(self.controls))
                controls_mask[0] = 0.0
                char_sets_mask = np.ones(len(self.char_sets))
                programs_mask = np.ones(len(self.programs))
                byte_chars_mask = np.ones(len(self.byte_chars))
                end_mask = np.zeros(1)
                return np.concatenate(
                    (controls_mask, char_sets_mask, programs_mask, byte_chars_mask, end_mask)
                )
            else:
                raise ("Unkown action")

        if action_type == "control":
            controls_mask = np.zeros(len(self.controls))
            if len(self.program_stack[-1][1]) > 1:
                if self.program_stack[-1][1][-2] not in self.controls:
                    controls_mask[-1] = 1.0
            char_sets_mask = np.ones(len(self.char_sets))
            programs_mask = np.ones(len(self.programs))
            programs_mask[-1] = 0.0
            byte_chars_mask = np.ones(len(self.byte_chars))
            end_mask = np.ones(1)
            return np.concatenate(
                (controls_mask, char_sets_mask, programs_mask, byte_chars_mask, end_mask)
            )

        if action_type in ["char_set", "char"]:
            controls_mask = np.ones(len(self.controls))
            char_sets_mask = np.ones(len(self.char_sets))
            programs_mask = np.ones(len(self.programs))
            byte_chars_mask = np.ones(len(self.byte_chars))
            end_mask = np.ones(1)
            return np.concatenate(
                (controls_mask, char_sets_mask, programs_mask, byte_chars_mask, end_mask)
            )

        if action_type == "return":
            controls_mask = np.ones(len(self.controls))
            char_sets_mask = np.ones(len(self.char_sets))
            programs_mask = np.ones(len(self.programs))
            byte_chars_mask = np.ones(len(self.byte_chars))
            end_mask = np.ones(1)
            return np.concatenate(
                (controls_mask, char_sets_mask, programs_mask, byte_chars_mask, end_mask)
            )

    def bracket_mode(self, action):
        if action != self.action_space - 1:
            action = action - len(self.controls) - len(self.char_sets) - len(self.programs)
            self.remaining_bracket_mask[action] = 0.0
            action = self.byte_chars[action]
            action_type = "continue"
            if action == b"-":
                action = b"\x5c\x2d"
            if len(self.program_stack[-1][1]) == 0 and action == b"^":
                action = b"\x5c\x5e"
            self.program_stack[-1][1].append(action)
        else:
            action_type = "return"

        if action_type == "continue":
            if self.program_stack[-1][1]:
                controls_mask = np.zeros(len(self.controls))
                char_sets_mask = np.zeros(len(self.char_sets))
                programs_mask = np.zeros(len(self.programs))
                end_mask = np.ones(1)
                return np.concatenate(
                    (
                        controls_mask,
                        char_sets_mask,
                        programs_mask,
                        self.remaining_bracket_mask,
                        end_mask,
                    )
                )
            else:
                action_type = "return"

        if action_type == "return":
            self.program_stack[-1][1].insert(0, b"[")
            self.program_stack[-1][1].append(b"]")
            self.program_stack[-2][1].append(self.program_stack[-1][1])
            # print(self.generate_regex(self.program_stack[-1][1]))
            self.program_stack.pop()
            controls_mask = np.ones(len(self.controls))
            char_sets_mask = np.ones(len(self.char_sets))
            programs_mask = np.ones(len(self.programs))
            byte_chars_mask = np.ones(len(self.byte_chars))
            end_mask = np.ones(1)
            return np.concatenate(
                (controls_mask, char_sets_mask, programs_mask, byte_chars_mask, end_mask)
            )

    def curly_bracket_mode(self, action):
        finish = False
        self.curly_values.append(action)

        if action < self.action_space - 2:
            for num in str(action).encode():
                self.program_stack[-1][1].append(bytes([num]))
        elif action == self.action_space - 2:
            self.program_stack[-1][1].append(b",")
        else:
            finish = True

        if len(self.curly_values) == 1:  # , or num
            if self.curly_values[-1] < self.action_space - 2:
                mask = np.zeros(self.action_space)
                mask[-1] = 1.0
                mask[-2] = 1.0
            else:
                mask = np.ones(self.action_space)
                mask[0] = 0.0
                mask[-1] = 0.0
                mask[-2] = 0.0
        if len(self.curly_values) == 2:  # num, or ,num
            if self.curly_values[-1] < self.action_space - 2:  # ,num
                finish = True
            else:
                mask = np.ones(self.action_space)
                mask[: self.curly_values[-2] + 2] = 0.0
                mask[-2] = 0.0
        if len(self.curly_values) == 3:  # num,num
            finish = True

        if finish:
            self.program_stack[-1][1].insert(0, b"{")
            self.program_stack[-1][1].append(b"}")
            self.program_stack[-2][1].append(self.program_stack[-1][1])
            # print(self.generate_regex(self.program_stack[-1][1]))
            self.program_stack.pop()
            controls_mask = np.zeros(len(self.controls))
            char_sets_mask = np.ones(len(self.char_sets))
            programs_mask = np.ones(len(self.programs))
            programs_mask[-1] = 0.0
            byte_chars_mask = np.ones(len(self.byte_chars))
            end_mask = np.ones(1)
            return np.concatenate(
                (controls_mask, char_sets_mask, programs_mask, byte_chars_mask, end_mask)
            )
        else:
            return mask

    def close_trace(self):
        while self.program_stack[-1][0] != "main":
            if self.program_stack[-1][0] == "parenthesis" and self.program_stack[-1][1]:
                self.program_stack[-1][1].insert(0, b"(")
                self.program_stack[-1][1].append(b")")
                self.program_stack[-2][1].append(self.program_stack[-1][1])
            elif self.program_stack[-1][0] == "bracket" and self.program_stack[-1][1]:
                self.program_stack[-1][1].insert(0, b"[")
                self.program_stack[-1][1].append(b"]")
                self.program_stack[-2][1].append(self.program_stack[-1][1])
            self.program_stack.pop()

    def generate_regex(self, l):
        regex = b""
        for elem in l:
            if isinstance(elem, list):
                regex += self.generate_regex(elem)
            else:
                regex += elem
        return regex

    def generate(self, length=np.inf):
        mask = self.reset()
        cpt = 0
        while isinstance(mask, np.ndarray) and cpt < length:
            action = np.random.choice(len(mask), None, p=mask / np.sum(mask))
            mask = self.step(action)
            cpt += 1
        if cpt == length:
            self.close_trace()
        return self.generate_regex(self.program_stack[0][1])
